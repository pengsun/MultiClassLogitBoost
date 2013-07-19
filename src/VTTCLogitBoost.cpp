#include "VTTCLogitBoost.hpp"

//#define  OUTPUT
#ifdef OUTPUT
#include <fstream>
std::ofstream os("output.txt");
#endif // OUTPUT

using namespace std;
using namespace cv;

// Implementation of VTTCLogitSplit
VTTCLogitSplit::VTTCLogitSplit()
{
  reset();
}

void VTTCLogitSplit::reset()
{
  var_idx_ = -1;
  threshold_ = FLT_MAX;
  subset_.reset();

  this_gain_ = -1;
  expected_gain_ = -1;
  left_node_gain_ = right_node_gain_ = -1;
}

// Implementation of KLogitNode
VTTCLogitNode::VTTCLogitNode(int _K)
{
  id_ = 0;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);
  fitvals_prev_.assign(_K,0);
}

VTTCLogitNode::VTTCLogitNode( int _id, int _K )
{
  id_ = _id;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);
  fitvals_prev_.assign(_K,0);
}

int VTTCLogitNode::calc_dir( float* _psample )
{
  float _val = *(_psample + split_.var_idx_);

  int dir = 0;
  if (split_.var_type_==VAR_CAT) {
    // TODO: raise an error
    /*
    int tmp = int(_val);
    dir = ( split_.subset_[tmp] == true ) ? (-1) : (+1);
    */
  }
  else { // split_.var_type_==VAR_NUM
    dir = (_val < split_.threshold_)? (-1) : (+1); 
  }

  return dir;
}
// Implementation of VTTCLogitSolver
//const double VTTCLogitSolver::EPS = 0.01;
const double VTTCLogitSolver::MAXGAMMA = 5.0;
VTTCLogitSolver::VTTCLogitSolver( VTTCLogitData*  _data)
{ 
  data_ = _data;
  
  int K = data_->data_cls_->get_class_count();
  mg_.assign(K, 0.0);
  h_.assign(K, 0.0);

}

void VTTCLogitSolver::update_internal( VecIdx& vidx )
{
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    update_internal_incre(idx);
  } // for it
}

void VTTCLogitSolver::update_internal_incre( int idx )
{
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[k] += (1-pik);
    else mg_[k] += (-pik);

    h_[k] += pik*(1-pik);
  }
}

void VTTCLogitSolver::update_internal_decre( int idx )
{
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[k] -= (1-pik);
    else mg_[k] -= (-pik);

    h_[k] -= pik*(1-pik);
  }

}

void VTTCLogitSolver::calc_gamma( const double* init_gamma, double lambda, double* gamma )
{
  int K = mg_.size();
  for (int kk = 0; kk < K; ++kk) {
    double smg = mg_[kk];
    double sh  = h_[kk];
     //if (sh <= 0) cv::error("VTTCLogitSolver::calc_gamma: Invalid Hessian.");
    if (sh == 0) sh = 1;

    // update node value based on L1 regularization (soft thresholding)
    double alpha = *(init_gamma+kk); // the current value (coordinate)
    double val1 = (-smg + lambda)/sh;
    double val2 = (-smg - lambda)/sh;
    double sgamma;
    if (alpha > val1)
      sgamma = alpha + (-val1);
    else if (alpha < val2)
      sgamma = alpha + (-val2);
    else
      sgamma = 0;

    // shrinkage
    //sgamma = 0.1 * sgamma;

    // capping
    double cap = sgamma;
    if (cap<-MAXGAMMA) cap = -MAXGAMMA;
    else if (cap>MAXGAMMA) cap = MAXGAMMA;
    *(gamma+kk) = cap;

  }
}

void VTTCLogitSolver::calc_gain( double& gain )
{
  gain = 0;
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double smg = mg_[k];
    double sh  = h_[k];
    if (sh == 0) sh = 1;

    gain += (smg*smg/sh);
  }
  gain = 0.5*gain;
}

// Implementation of VTTCLogitTree::Param
VTTCLogitTree::Param::Param()
{
  max_leaves_ = 2;
  node_size_ = 5;
  lambda_ = 0.0;
}
// Implementation of VTTCLogitTree
void VTTCLogitTree::split( VTTCLogitData* _data )
{
  // clear
  clear();
  K_ = _data->data_cls_->get_class_count();

  // root node
  creat_root_node(_data);
  candidate_nodes_.push(&nodes_.front());
  VTTCLogitNode* root = candidate_nodes_.top(); 
  find_best_candidate_split(root, _data);
  int nleaves = 1;

  // split recursively
  while ( nleaves < param_.max_leaves_ &&
          !candidate_nodes_.empty() )
  {
    VTTCLogitNode* cur_node = candidate_nodes_.top(); // the most prior node
    candidate_nodes_.pop();
    --nleaves;

    if (!can_split_node(cur_node)) { // can not split, make it a leaf
      ++nleaves;
      continue;
    }

    split_node(cur_node,_data);
    // release memory: no longer used in later splitting
    VecIdx tmp;
    tmp.swap(cur_node->sample_idx_); 

    // find best split for the two newly created nodes
    find_best_candidate_split(cur_node->left_, _data);
    candidate_nodes_.push(cur_node->left_);
    ++nleaves;

    find_best_candidate_split(cur_node->right_, _data);
    candidate_nodes_.push(cur_node->right_);
    ++nleaves;
  }
}

void VTTCLogitTree::fit( VTTCLogitData* _data )
{
  // fitting node data for each leaf
  std::list<VTTCLogitNode>::iterator it;
  for (it = nodes_.begin(); it != nodes_.end(); ++it) {
    VTTCLogitNode* nd = &(*it);

    if (nd->left_!=0) { // not a leaf
      continue;
    } 

    fit_node(nd,_data);

    // do NOT remove example index for this leaf
#if 0
    // release memory: no longer used in later splitting
    VecIdx tmp;
    tmp.swap(nd->sample_idx_);
#endif
  }
}


VTTCLogitNode* VTTCLogitTree::get_node( float* _sample)
{
  VTTCLogitNode* cur_node = &(nodes_.front());
  while (true) {
    if (cur_node->left_==0) break; // leaf reached 

    int dir = cur_node->calc_dir(_sample);
    VTTCLogitNode* next = (dir==-1) ? (cur_node->left_) : (cur_node->right_);
    cur_node = next;
  }
  return cur_node;
}
void VTTCLogitTree::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void VTTCLogitTree::predict( float* _sample, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update all K classes
  VTTCLogitNode* nd;
  nd = get_node(_sample);

  for (int k = 0; k < K_; ++k) {
    *(_score + k) = static_cast<float>( nd->fitvals_[k] );
  }

}

void VTTCLogitTree::clear()
{
  nodes_.clear();
  candidate_nodes_.~priority_queue();
}

void VTTCLogitTree::creat_root_node( VTTCLogitData* _data )
{
  nodes_.push_back(VTTCLogitNode(0,K_));
  VTTCLogitNode* root = &(nodes_.back());

  // samples in node
  int N = _data->data_cls_->X.rows;
  root->sample_idx_.resize(N);
  for (int i = 0; i < N; ++i) {
    root->sample_idx_[i] = i;
  }

  // loss
  this->calc_gain(root, _data);
}

bool VTTCLogitTree::find_best_candidate_split( VTTCLogitNode* _node, VTTCLogitData* _data )
{
  bool found_flag = false;
  MLData* data_cls = _data->data_cls_;
  int nvar = data_cls->X.cols;


  cvb_split_.reset(); cb_split_.reset();
  for (int ivar = 0; ivar < nvar; ++ivar) {
    bool ret;
    if (data_cls->var_type[ivar] == VAR_CAT) {
      // TODO: raise an error
    }
    else { // VAR_NUM
      ret = find_best_split_num_var(_node, _data,ivar);
    }

    if (!ret) continue; // nothing found

    if (cvb_split_.expected_gain_ > cb_split_.expected_gain_) {
      found_flag = true;
      cb_split_ = cvb_split_;   
    }

  }

  if (found_flag) {
    _node->split_ = cb_split_;
    return true;
  }
  else {
    _node->split_.var_idx_ = -1;
    return false;
  }
}

bool VTTCLogitTree::find_best_split_num_var( 
  VTTCLogitNode* _node, VTTCLogitData* _data, int _ivar)
{
  VecIdx node_sample_si;
  MLData* data_cls = _data->data_cls_;
  make_node_sorted_idx(_node,data_cls,_ivar,node_sample_si);
  int ns = node_sample_si.size();
  CV_Assert(ns >= 1);

  // initialize
  VTTCLogitSolver sol_left(_data), sol_right(_data);
  sol_right.update_internal(node_sample_si);

  // scan each possible split 
  double best_gain = -1, best_gain_left = -1, best_gain_right = -1;
  int best_i = -1;
  for (int i = 0; i < ns-1; ++i) {  // ** excluding null and all **
    int idx = node_sample_si[i];
    sol_left.update_internal_incre(idx);
    sol_right.update_internal_decre(idx);

    // skip if overlap
    int idx1 = idx;
    float x1 = data_cls->X.at<float>(idx1, _ivar);
    int idx2 = node_sample_si[i+1];
    float x2 = data_cls->X.at<float>(idx2, _ivar);
    if (x1==x2) continue; // overlap

    // check left & right
    double gL;
    sol_left.calc_gain(gL);
    double gR;
    sol_right.calc_gain(gR);

    double g = gL + gR;
    if (g > best_gain) {
      best_i = i;
      best_gain = g;
      best_gain_left = gL; best_gain_right = gR;
    } // if
  } // for i

  // set output
  return set_best_split_num_var(
    _node, data_cls, _ivar,
    node_sample_si,
    best_i, best_gain, best_gain_left, best_gain_right);
}

void VTTCLogitTree::make_node_sorted_idx( VTTCLogitNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node )
{
  VecIdx16 sam_idx16;
  VecIdx32 sam_idx32;
  if (_data->is_idx16) 
    sam_idx16 = _data->var_num_sidx16[_ivar];
  else
    sam_idx32 = _data->var_num_sidx32[_ivar];

  // mask for samples in _node
  int N = _data->X.rows;
  vector<bool> mask(N,false); 
  int nn = _node->sample_idx_.size();
  for (int i = 0; i < nn; ++i) {
    int idx = _node->sample_idx_[i];
    mask[idx] = true;
  }

  // copy the sorted indices for samples in _node
  sorted_idx_node.reserve(nn);
  if (_data->is_idx16) {
    for (int i = 0; i < N; ++i) {
      int ix = int(sam_idx16[i]);
      if (mask[ix]) 
        sorted_idx_node.push_back( ix );
    }
  }
  else {
    for (int i = 0; i < N; ++i) {
      int ix = int(sam_idx32[i]);
      if (mask[ix])
        sorted_idx_node.push_back(ix);
    }
  }  
}

bool VTTCLogitTree::set_best_split_num_var( 
  VTTCLogitNode* _node, MLData* _data, int _ivar, 
  VecIdx& node_sample_si, 
  int best_i, double best_gain, double best_gain_left, double best_gain_right )
{
  if (best_i==-1) return false; // fail to find...

  // set gains
  double this_gain = _node->split_.this_gain_;
  cvb_split_.this_gain_ = this_gain;
  cvb_split_.expected_gain_ = best_gain - this_gain; 
  cvb_split_.left_node_gain_ = best_gain_left;
  cvb_split_.right_node_gain_ = best_gain_right;

  // set split
  cvb_split_.var_idx_ = _ivar;
  cvb_split_.var_type_ = _data->var_type[_ivar]; 
  int idx1 = node_sample_si[best_i];
  float x1 = _data->X.at<float>(idx1, _ivar);
  int idx2 = node_sample_si[best_i+1];
  float x2 = _data->X.at<float>(idx2, _ivar);
  if (x2>x1)
    cvb_split_.threshold_ = (x1+x2)/2;
  else
    return false; // all samples overlap, fail to split...

  return true;  
}

bool VTTCLogitTree::can_split_node( VTTCLogitNode* _node )
{
  bool flag = true;
  int nn = _node->sample_idx_.size();
  int idx = _node->split_.var_idx_;
  return (nn > param_.node_size_    && // large enough node size
          idx != -1);                  // has candidate split  
}

bool VTTCLogitTree::split_node( VTTCLogitNode* _node, VTTCLogitData* _data )
{
  // create left and right node
  VTTCLogitNode tmp1(nodes_.size(), K_);
  nodes_.push_back(tmp1);
  _node->left_ = &(nodes_.back());
  _node->left_->parent_ = _node;

  VTTCLogitNode tmp2(nodes_.size(), K_);
  // 
  nodes_.push_back(tmp2);
  _node->right_ = &(nodes_.back());
  _node->right_->parent_ = _node;

  // send each sample to left/right node
  int nn = _node->sample_idx_.size();
  CV_Assert(_node->split_.var_idx_>-1);
  MLData* data_cls = _data->data_cls_;
  for (int i = 0; i < nn; ++i) {
    int idx = _node->sample_idx_[i];

    float* p = (float*)data_cls->X.ptr(idx);
    int dir = _node->calc_dir(p);
    if (dir == -1) 
      _node->left_->sample_idx_.push_back(idx);
    else 
      _node->right_->sample_idx_.push_back(idx);
  }

  // initialize the node gain
  this->calc_gain(_node->left_, _data);
  this->calc_gain(_node->right_, _data);

#ifdef OUTPUT
  os << "id = " << _node->id_ << ", ";
  os << "ivar = " << _node->split_.var_idx_ << ", ";
  os << "th = " << _node->split_.threshold_ << ", ";
  os << "idL = " << _node->left_->id_ << ", " 
    << "idR = " << _node->right_->id_ << ", ";
  os << "n = " << _node->sample_idx_.size() << ", ";
  os << "this_gain = " << _node->split_.this_gain_ << ", ";
  os << "exp_gain = " << _node->split_.expected_gain_ << endl;
#endif // OUTPUT

  return true;
}

void VTTCLogitTree::calc_gain(VTTCLogitNode* _node, VTTCLogitData* _data)
{
  VTTCLogitSolver sol(_data);
  sol.update_internal(_node->sample_idx_);
  double gain;
  sol.calc_gain(gain);
  _node->split_.this_gain_ = gain;
}

void VTTCLogitTree::fit_node( VTTCLogitNode* _node, VTTCLogitData* _data )
{
  int nn = _node->sample_idx_.size();
  CV_Assert(nn>0);

  VTTCLogitSolver sol(_data);

  sol.update_internal(_node->sample_idx_);

  // cache node values before updating
  _node->fitvals_prev_ = _node->fitvals_;
  sol.calc_gamma(&(_node->fitvals_prev_[0]), this->param_.lambda_, 
                 &(_node->fitvals_[0]) );

#ifdef OUTPUT
  //os << "id = " << _node->id_ << "(ter), ";
  //os << "n = " << _node->sample_idx_.size() << ", ";
  //os << "cls = (" << _node->cls1_ << ", " << _node->cls2_ << "), ";

  //// # of cls1 and cls2
  //int ncls1 = 0, ncls2 = 0;
  //for (int i = 0; i < _node->sample_idx_.size(); ++i) {
  //  int ix = _node->sample_idx_[i];
  //  int k = static_cast<int>( _data->data_cls_->Y.at<float>(ix) );
  //  if ( k == cls1) ncls1++;
  //  if ( k == cls2) ncls2++;
  //}
  //// os << "ncls1 = " << ncls1 << ", " << "ncls2 = " << ncls2 << ", ";

  //// min, max of p
  //vector<double> pp1, pp2;
  //for (int i = 0; i < _node->sample_idx_.size(); ++i) {
  //  int ix = _node->sample_idx_[i];
  //  double* ptr = _data->p_->ptr<double>(ix);
  //  int k = static_cast<int>( _data->data_cls_->Y.at<float>(ix) );
  //  if ( k == cls1) {
  //    pp1.push_back( *(ptr+k) );
  //  }
  //  if ( k == cls2) {
  //    pp2.push_back( *(ptr+k) );
  //  }
  //}

  //vector<double>::iterator it;
  //double pp1max, pp1min;
  //if (!pp1.empty()) {
  //  it = std::max(pp1.begin(), pp1.end());
  //  if (it==pp1.end()) it = pp1.begin();
  //  pp1max = *it;
  //  it = std::min(pp1.begin(), pp1.end());
  //  if (it==pp1.end()) it = pp1.begin();
  //  pp1min = *it;
  //  //os << "pp1max = " << pp1max << ", " << "pp1min = " << pp1min << ", ";
  //}
  //
  //double pp2max, pp2min;
  //if (!pp2.empty()) {
  //  it = std::max(pp2.begin(), pp2.end());
  //  if (it==pp2.end()) it = pp2.begin();
  //  pp2max = *it;
  //  it = std::min(pp2.begin(), pp2.end());
  //  if (it==pp2.end()) it = pp2.begin();
  //  pp2min = *it;
  //  //os << "pp2max = " << pp2max << ", " << "pp2min = " << pp2min << ", "; 
  //}

  //if (ncls1==0) os << "cls1 (n = 0), ";
  //else 
  //  os << "cls1 (n = " << ncls1 << ", pmax = " << pp1max
  //     << ", pmin = " << pp1min << "), ";

  //if (ncls2==0) os << "cls2 (n = 0), ";
  //else
  //  os << "cls2 (n = " << ncls2 << ", pmax = " << pp2max
  //  << ", pmin = " << pp2min << "), ";

  //os << "gamma = " << _node->fit_val_ << endl;
#endif // OUTPUT
}



void VTTCLogitTree::predict_prev_val( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict_prev_val(p,score);
  }
}

void VTTCLogitTree::predict_prev_val( float* _sample, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update all K classes
  VTTCLogitNode* nd;
  nd = get_node(_sample);

  for (int k = 0; k < K_; ++k) {
    *(_score + k) = static_cast<float>( nd->fitvals_prev_[k] );
  }

}

// Implementation of VTTCLogitBoost::Param
VTTCLogitBoost::Param::Param()
{
  T = 2;
  lambda = 0.0;
  J = 4;
  ns = 1;
  TRound = 6;
}
// Implementation of VTTCLogitBoost
const double VTTCLogitBoost::EPS_LOSS = 1e-14;
const double VTTCLogitBoost::MAX_F = 100;
void VTTCLogitBoost::train( MLData* _data )
{
  train_init(_data);

  for (int t = 0; t < param_.T; ++t) {
#ifdef OUTPUT
    os << "t = " << t << endl;
#endif // OUTPUT
    trees_[t].split(&logitdata_);
#if 0
    trees_[t].fit(&logitdata_);
#endif

    // Totally corrective: cyclically revisit and update all tree leveas
    update_all_tree_leaves(t);

    // update loss & gradients
    calc_loss(_data);
    calc_loss_iter(t);
    calc_grad(t);

#ifdef OUTPUT
    //os << "loss = " << L_iter_.at<double>(t) << endl;
#endif // OUTPUT

    NumIter_ = t + 1;
    if ( should_stop(t) ) break;
  } // for t

}

void VTTCLogitBoost::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void VTTCLogitBoost::predict( float* _sapmle, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  } // for k

  // sum of tree
  vector<float> s(K_);
  for (int t = 0; t < NumIter_; ++t) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (s[k]);
    } // for k
  } // for t
}

void VTTCLogitBoost::predict( MLData* _data, int _Tpre )
{
  // trees to be used
  if (_Tpre > NumIter_) _Tpre = NumIter_;
  if (_Tpre < 1) _Tpre = 1; // _Tpre in [1,T]
  if (Tpre_beg_ > _Tpre) Tpre_beg_ = 0;

  // initialize predicted score
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  // initialize internal score if necessary
  if (Tpre_beg_ == 0) {
    Fpre_.create(N,K);
    Fpre_ = 0;
  }

  // for each sample
  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score, _Tpre);

    // update score and internal score Fpre_
    double* pp = Fpre_.ptr<double>(i);
    for (int k = 0; k < K; ++k) {
      *(score+k) += *(pp+k);
      *(pp+k) = *(score+k);
    }
  }

  // Set the new beginning tree
  Tpre_beg_ = _Tpre;
}

void VTTCLogitBoost::predict( float* _sapmle, float* _score, int _Tpre )
{
  // IMPORTANT: caller should assure the validity of _Tpre

  // initialize 
  for (int k = 0; k < K_; ++k) 
    *(_score+k) = 0;

  // sum of tree
  vector<float> s(K_);
  for (int t = Tpre_beg_; t < _Tpre; ++t ) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (s[k]);
    }
  }

}


int VTTCLogitBoost::get_class_count()
{
  return K_;
}

int VTTCLogitBoost::get_num_iter()
{
  return NumIter_;
}

#if 0
double VTTCLogitBoost::get_train_loss()
{
  if (NumIter_<1) return DBL_MAX;
  return L_iter_.at<double>(NumIter_-1);
}
#endif
void VTTCLogitBoost::train_init( MLData* _data )
{
  // class count
  K_ = _data->get_class_count();

  // F, p
  int N = _data->X.rows;
  F_.create(N,K_); 
  F_ = 1;
  p_.create(N,K_); 
  update_p();

  // Loss
  L_.create(N,1);
  calc_loss(_data);
  L_iter_.create(param_.T,1);

  // iteration for training
  NumIter_ = 0;

  // logitData
  logitdata_.data_cls_ = _data;
  logitdata_.p_ = &p_;
  logitdata_.L_ = &L_;

  // trees
  trees_.clear();
  trees_.resize(param_.T);
  for (int t = 0; t < param_.T; ++t) {
    trees_[t].param_.max_leaves_ = param_.J;
    trees_[t].param_.node_size_ = param_.ns;
    trees_[t].param_.lambda_ = param_.lambda;
  }

  // gradient/delta
  abs_grad_.clear();
  abs_grad_.resize(param_.T);

  // for prediction
  Tpre_beg_ = 0;
}

void VTTCLogitBoost::update_F_using_tree(int t)
{
  int N = logitdata_.data_cls_->X.rows;
  vector<float> f(K_), f_prev(K_); // current, previous tree node values
  for (int i = 0; i < N; ++i) {
    float *psample = logitdata_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);
    trees_[t].predict_prev_val(psample,&f_prev[0]);

    double* pF = F_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(pF+k) -= (f_prev[k]);
      *(pF+k) += (f[k]);
      // MAX cap
      if (*(pF+k) > MAX_F) *(pF+k) = MAX_F; // TODO: make the threshold a constant variable
    } // for k
  } // for i
}

void VTTCLogitBoost::update_p()
{
  int N = F_.rows;
  int K = K_;
  std::vector<double> tmpExpF(K);

  for (int n = 0; n < N; ++n) {
    double tmpSumExpF = 0;
    double* ptrF = F_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      double Fnk = *(ptrF + k);
      double tmp = exp(Fnk);
      tmpExpF[k] = tmp;
      tmpSumExpF += tmp;
    } // for k

    double* ptrp = p_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      // TODO: does it make any sense??
      if (tmpSumExpF==0) tmpSumExpF = 1;
      *(ptrp + k) = double( tmpExpF[k]/tmpSumExpF );
    } // for k
  }// for n  
}

#if 0
bool VTTCLogitBoost::should_stop( int t )
{
  int N = F_.rows;
  //double peps = exp(MIN_F-1); // min p <--> MIN_F
  double delta = 0;
    
  for (int i = 0; i < N; ++i) {
    double* ptr_pi = p_.ptr<double>(i);
    int yi = int( logitdata_.data_cls_->Y.at<float>(i) );

    for (int k = 0; k < K_; ++k) {
      double pik = *(ptr_pi+k);
      if (yi==k) delta += std::abs( 1-pik );
      else       delta += std::abs( -pik );    
    }
  }
  
  abs_grad_[t] = delta;
  if ( delta < (2*N*K_*1e-3) ) // TODO: make the threshold a constant variable
    return true;
  else
    return false;
}
#endif
void VTTCLogitBoost::calc_loss( MLData* _data )
{
  const double PMIN = 0.0001;
  int N = _data->X.rows;
  for (int i = 0; i < N; ++i) {
    int yi = int( _data->Y.at<float>(i) );
    double* ptr = p_.ptr<double>(i);
    double pik = *(ptr + yi);

    if (pik<PMIN) pik = PMIN;
    L_.at<double>(i) = (-log(pik));
  }
}

void VTTCLogitBoost::calc_loss_iter( int t )
{
  double sum = 0;
  int N = L_.rows;
  for (int i = 0; i < N; ++i) 
    sum += L_.at<double>(i);

  L_iter_.at<double>(t) = sum;
}

bool VTTCLogitBoost::should_stop( int t )
{
  double loss = L_iter_.at<double>(t);
  return ( (loss<EPS_LOSS) ? true : false );
}

void VTTCLogitBoost::calc_grad( int t )
{
  int N = F_.rows;
  double delta = 0;
    
  for (int i = 0; i < N; ++i) {
    double* ptr_pi = p_.ptr<double>(i);
    int yi = int( logitdata_.data_cls_->Y.at<float>(i) );

    for (int k = 0; k < K_; ++k) {
      double pik = *(ptr_pi+k);
      if (yi==k) delta += std::abs( 1-pik );
      else       delta += std::abs( -pik );    
    }
  }
  
  abs_grad_[t] = delta;
}

void VTTCLogitBoost::update_all_tree_leaves(int t)
{
  // at each round, cyclize each tree
  for (int r = 0; r < param_.TRound; ++r) {
    // for each tree, update its leaf values
    for (int itree = t; itree >= 0; --itree) {
      trees_[itree].fit(&logitdata_);

      // update working variables. note that each node's examples are disjoint, so
      // we can do the updating late
      update_F_using_tree(itree);
      update_p();
    }
  }

}