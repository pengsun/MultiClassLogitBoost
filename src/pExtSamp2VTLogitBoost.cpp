#include "pExtSamp2VTLogitBoost.hpp"
#include <functional>
#include <numeric>

//#define  OUTPUT
#ifdef OUTPUT
#include <fstream>
std::ofstream os("output.txt");
#endif // OUTPUT

using namespace std;
using namespace cv;

static cv::RNG THE_RNG;

namespace {
  struct IdxGreater {
    IdxGreater (VecDbl *_v) { v_ = _v;};

    bool operator () (int i1, int i2) {
      return ( v_->at(i1) > v_->at(i2) );
    };
    VecDbl *v_;
  };

  struct IdxMatGreater {
    IdxMatGreater (cv::Mat_<double> *_v) { v_ = _v;};

    bool operator () (int i1, int i2) {
      return ( v_->at<double>(i1) > v_->at<double>(i2) );
    };
    cv::Mat_<double> *v_;
  };

  void release_VecDbl (VecDbl &in) {
    VecDbl tmp;
    tmp.swap(in);
  }

  void uniform_subsample_ratio (int N, double ratio, VecIdx &ind) {
    ind.clear();
    // sample without replacement
    for (int i = 0; i < N; ++i) {
      double num = THE_RNG.uniform((double)0.0, (double)1.0);
      if (num < ratio) ind.push_back(i);
    }

    // pick a random one if empty
    if (ind.empty()) { 
      int iii = THE_RNG.uniform((int)0, (int)(N-1));
      ind.push_back(iii);
    }
  }

  void weight_trim_ratio (cv::Mat_<double>& w, double ratio, VecIdx &ind) {
    int N = w.rows;

    // sorting the index in descending order,
    VecIdx wind(N);
    for (int i = 0; i < N; ++i) wind[i] = i;
    std::sort (wind.begin(),wind.end(), IdxMatGreater(&w));

    // normalization factor
    double Z = std::accumulate(w.begin(),w.end(), 0.0);

    // trim
    ind.clear();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      int ii = wind[i];
      sum += w.at<double>(ii)/Z;

      if (sum < ratio) ind.push_back(ii);
    }

    // pick a random one if empty
    if (ind.empty()) { 
      int iii = THE_RNG.uniform((int)0, (int)(N-1));
      ind.push_back(iii);
    }
  }


}

// Implementation of pExtSamp2VTSplit
pExtSamp2VTSplit::pExtSamp2VTSplit()
{
  reset();
}

void pExtSamp2VTSplit::reset()
{
  var_idx_ = -1;
  threshold_ = FLT_MAX;
  subset_.reset();

  this_gain_ = -1;
  expected_gain_ = -1;
  left_node_gain_ = right_node_gain_ = -1;
}


// Implementation of pExtSamp2VTSolver
//const double pExtSamp2VTSolver::EPS = 0.01;
const double pExtSamp2VTSolver::MAXGAMMA = 5.0;
pExtSamp2VTSolver::pExtSamp2VTSolver( pExtSamp2VTData* _data, VecIdx* _ci)
{ 
  set_data(_data, _ci);
}

void pExtSamp2VTSolver::set_data( pExtSamp2VTData* _data, VecIdx* _ci)
{ 
  data_ = _data;
  ci_ = _ci;

  int KK = _ci->size();
  mg_.assign(KK, 0.0);
  h_.assign(KK, 0.0);
}

void pExtSamp2VTSolver::update_internal( VecIdx& vidx )
{
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    update_internal_incre(idx);
  } // for it
}

void pExtSamp2VTSolver::update_internal_incre( int idx )
{
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    int k = this->ci_->at(kk);
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[kk] += (1-pik);
    else       mg_[kk] += (-pik);

    h_[kk] += pik*(1-pik);
  }
}

void pExtSamp2VTSolver::update_internal_decre( int idx )
{
  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    int k = this->ci_->at(kk);
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[kk] -= (1-pik);
    else       mg_[kk] -= (-pik);

    h_[kk] -= pik*(1-pik);
  }
}

void pExtSamp2VTSolver::calc_gamma( double *gamma)
{
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    double smg = mg_[kk];
    double sh  = h_[kk];
    //if (sh <= 0) cv::error("pExtSamp2VTSolver::calc_gamma: Invalid Hessian.");
    if (sh == 0) sh = 1;

    double sgamma = smg/sh;
    double cap = sgamma;
    if (cap<-MAXGAMMA) cap = -MAXGAMMA;
    else if (cap>MAXGAMMA) cap = MAXGAMMA;

    // do the real updating
    int k = this->ci_->at(kk);
    *(gamma+k) = cap;

  }
}

void pExtSamp2VTSolver::calc_gain( double& gain )
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


// Implementation of pExtSamp2VTNode
pExtSamp2VTNode::pExtSamp2VTNode(int _K)
{
  id_ = 0;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);
}

pExtSamp2VTNode::pExtSamp2VTNode( int _id, int _K )
{
  id_ = _id;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);
}

int pExtSamp2VTNode::calc_dir( float* _psample )
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

// Implementation of pExt2VT_best_split_finder (helper class)
pExt2VT_best_split_finder::pExt2VT_best_split_finder(pExtSamp2VTTree *_tree, 
  pExtSamp2VTNode *_node, pExtSamp2VTData *_data)
{
  this->tree_ = _tree;
  this->node_ = _node;
  this->data_ = _data;

  this->cb_split_.reset();
}

pExt2VT_best_split_finder::pExt2VT_best_split_finder (const pExt2VT_best_split_finder &f, cv::Split)
{
  this->tree_ = f.tree_;
  this->node_ = f.node_;
  this->data_ = f.data_;
  this->cb_split_ = f.cb_split_;
}

void pExt2VT_best_split_finder::operator() (const cv::BlockedRange &r)
{

  // for each variable, find the best split
  for (int ii = r.begin(); ii != r.end(); ++ii) {
    int vi = this->tree_->sub_fi_[ii];

    pExtSamp2VTSplit the_split;
    the_split.reset();
    bool ret;
    ret = tree_->find_best_split_num_var(node_, data_, vi, 
      the_split);

    // update the cb_split (currently best split)
    if (!ret) continue; // nothing found
    if (the_split.expected_gain_ > cb_split_.expected_gain_) {
      cb_split_ = the_split;
    } // if
  } // for vi
}

void pExt2VT_best_split_finder::join (pExt2VT_best_split_finder &rhs)
{
  if ( rhs.cb_split_.expected_gain_ > (this->cb_split_.expected_gain_) ) {
    (this->cb_split_) = (rhs.cb_split_);
  }
}

// Implementation of pExtSamp2VTTree::Param
pExtSamp2VTTree::Param::Param()
{
  max_leaves_ = 2;
  node_size_ = 5;
  ratio_si_ = ratio_fi_ = 0.6;
  ratio_ci_ = 0.8;
}

// Implementation of pExtSamp2VTTree
void pExtSamp2VTTree::split( pExtSamp2VTData* _data )
{
  // clear
  clear();
  K_ = _data->data_cls_->get_class_count();

  // do subsampling for samples, features and classes
  subsample(_data);

  // root node
  creat_root_node(_data);
  candidate_nodes_.push(&nodes_.front());
  pExtSamp2VTNode* root = candidate_nodes_.top(); 
  find_best_candidate_split(root, _data);
  int nleaves = 1;

  // split recursively
  while ( nleaves < param_.max_leaves_ &&
          !candidate_nodes_.empty() )
  {
    pExtSamp2VTNode* cur_node = candidate_nodes_.top(); // the most prior node
    candidate_nodes_.pop();
    --nleaves;

    if (!can_split_node(cur_node)) { // can not split, make it a leaf
      ++nleaves;
      continue;
    }

    split_node(cur_node,_data);
    VecIdx tmp;
    tmp.swap(cur_node->sample_idx_); 
    // release memory.
    // no longer used in later splitting
    release_VecDbl( cur_node->sol_this_.mg_ );
    release_VecDbl( cur_node->sol_this_.h_ );

    // find best split for the two newly created nodes
    find_best_candidate_split(cur_node->left_, _data);
    candidate_nodes_.push(cur_node->left_);
    ++nleaves;

    find_best_candidate_split(cur_node->right_, _data);
    candidate_nodes_.push(cur_node->right_);
    ++nleaves;
  }
}

void pExtSamp2VTTree::fit( pExtSamp2VTData* _data )
{
  // fitting node data for each leaf
  std::list<pExtSamp2VTNode>::iterator it;
  for (it = nodes_.begin(); it != nodes_.end(); ++it) {
    pExtSamp2VTNode* nd = &(*it);

    if (nd->left_!=0) { // not a leaf
      continue;
    } 

    fit_node(nd,_data);

    // release memory.
    // no longer used in later splitting
    VecIdx tmp;
    tmp.swap(nd->sample_idx_);
    release_VecDbl( nd->sol_this_.mg_ );
    release_VecDbl( nd->sol_this_.h_ );
  }
}


pExtSamp2VTNode* pExtSamp2VTTree::get_node( float* _sample)
{
  pExtSamp2VTNode* cur_node = &(nodes_.front());
  while (true) {
    if (cur_node->left_==0) break; // leaf reached 

    int dir = cur_node->calc_dir(_sample);
    pExtSamp2VTNode* next = (dir==-1) ? (cur_node->left_) : (cur_node->right_);
    cur_node = next;
  }
  return cur_node;
}

void pExtSamp2VTTree::predict( MLData* _data )
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

void pExtSamp2VTTree::predict( float* _sample, float* _score )
{
  // initialize all the *K* classes
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update all the K classes...
  pExtSamp2VTNode* nd = get_node(_sample);
  for (int k = 0; k < K_; ++k) {
    float val = static_cast<float>( nd->fitvals_[k] );
    *(_score + k) = val;
  }

}


void pExtSamp2VTTree::subsample(pExtSamp2VTData* _data)
{
  // subsample samples
  int NS = _data->data_cls_->X.rows;
#if 0
  uniform_subsample_ratio(NS,this->param_.ratio_si_, this->sub_si_);
#endif
  weight_trim_ratio ( *(_data->g_), this->param_.ratio_si_, this->sub_si_);

  // subsample features
  int NF = _data->data_cls_->X.cols;
  uniform_subsample_ratio(NF,this->param_.ratio_fi_, this->sub_fi_);

  // subsample classes via gradients
#if 0
  // subsample classes
  int NC = _data->data_cls_->get_class_count();
  uniform_subsample_ratio(NC,this->param_.ratio_ci_, this->sub_ci_);
  if (this->sub_ci_.size() < 2) { // at least size = 1
    while (true) {
      int n = THE_RNG.uniform((int)0, (int)(K_-1));
      if ( n != this->sub_ci_[0]) {
        this->sub_ci_.push_back(n);
        break;
      }
    } // while
  } // if
#endif
#if 1
  // all sample index
  VecIdx all_si(NS);
  for (int i = 0; i < NS; ++i) all_si[i] = i;
  // all class index
  int NC = _data->data_cls_->get_class_count();
  VecIdx all_ci(NC);
  for (int i = 0; i < NC; ++i) all_ci[i] = i;
  // get minus gradient via solver
  pExtSamp2VTSolver tmp_sol(_data, &all_ci);
  tmp_sol.update_internal(all_si);
  VecDbl absg = tmp_sol.mg_;
  for (int i = 0; i < absg.size(); ++i) absg[i] = abs((double)absg[i]);
  // sorting the index in descending order, 
  VecIdx absgind(absg.size());
  for (int i = 0; i < absg.size(); ++i) 
    absgind[i] = i;
  std::sort (absgind.begin(),absgind.end(), IdxGreater(&absg));
  // select the first #*ratio classes
  int Nthre = int( double(NC)*this->param_.ratio_ci_ );
  if (Nthre<2) Nthre = 2;
  sub_ci_.resize(Nthre);
  for (int i = 0; i < Nthre; ++i)
    sub_ci_[i] = absgind[i];
#endif
  
}

void pExtSamp2VTTree::clear()
{
  nodes_.clear();
  candidate_nodes_.~priority_queue();
}

void pExtSamp2VTTree::creat_root_node( pExtSamp2VTData* _data )
{
  nodes_.push_back(pExtSamp2VTNode(0,K_));
  pExtSamp2VTNode* root = &(nodes_.back());

  // samples in node
  int NN = this->sub_si_.size();
  root->sample_idx_.resize(NN);
  for (int ii = 0; ii < NN; ++ii) {
    int ind = sub_si_[ii];
    root->sample_idx_[ii] = ind;
  }

  // initialize solver
  root->sol_this_.set_data(_data, &(this->sub_ci_));
  root->sol_this_.update_internal(root->sample_idx_);

  // loss
  this->calc_gain(root, _data);
}

bool pExtSamp2VTTree::find_best_candidate_split( pExtSamp2VTNode* _node, pExtSamp2VTData* _data )
{
  // subsample the features whenever finding the best split
  int NF = _data->data_cls_->X.cols;
  uniform_subsample_ratio(NF,this->param_.ratio_fi_, this->sub_fi_);

  // the range (beginning/ending variable)
  int nsubvar = this->sub_fi_.size();
  cv::BlockedRange br(0,nsubvar,1);

  // do the search in parallel
  pExt2VT_best_split_finder bsf(this,_node,_data);
  cv::parallel_reduce(br, bsf);

  // update node's split
  _node->split_ = bsf.cb_split_;
  return true; // TODO: Check if this is reasonable

}

bool pExtSamp2VTTree::find_best_split_num_var( 
  pExtSamp2VTNode* _node, pExtSamp2VTData* _data, int _ivar, pExtSamp2VTSplit &cb_split)
{
  VecIdx node_sample_si;
  MLData* data_cls = _data->data_cls_;
  make_node_sorted_idx(_node,data_cls,_ivar,node_sample_si);
  int ns = node_sample_si.size();

#if 0
  CV_Assert(ns >= 1);
#endif
#if 1
  if (ns < 1) return false;
#endif

  // initialize
  pExtSamp2VTSolver sol_left(_data, &(this->sub_ci_)), sol_right = _node->sol_this_;

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
    best_i, best_gain, best_gain_left, best_gain_right,
    cb_split);
}

void pExtSamp2VTTree::make_node_sorted_idx( pExtSamp2VTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node )
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

bool pExtSamp2VTTree::set_best_split_num_var( 
  pExtSamp2VTNode* _node, MLData* _data, int _ivar, 
  VecIdx& node_sample_si, 
  int best_i, double best_gain, double best_gain_left, double best_gain_right,
  pExtSamp2VTSplit &cb_split)
{
  if (best_i==-1) return false; // fail to find...

  // set gains
  double this_gain = _node->split_.this_gain_;
  cb_split.this_gain_ = this_gain;
  cb_split.expected_gain_ = best_gain - this_gain; 
  cb_split.left_node_gain_ = best_gain_left;
  cb_split.right_node_gain_ = best_gain_right;

  // set split
  cb_split.var_idx_ = _ivar;
  cb_split.var_type_ = _data->var_type[_ivar]; 
  int idx1 = node_sample_si[best_i];
  float x1 = _data->X.at<float>(idx1, _ivar);
  int idx2 = node_sample_si[best_i+1];
  float x2 = _data->X.at<float>(idx2, _ivar);
  if (x2>x1)
    cb_split.threshold_ = (x1+x2)/2;
  else
    return false; // all samples overlap, fail to split...

  return true;  
}

bool pExtSamp2VTTree::can_split_node( pExtSamp2VTNode* _node )
{
  bool flag = true;
  int nn = _node->sample_idx_.size();
  int idx = _node->split_.var_idx_;
  return (nn > param_.node_size_    && // large enough node size
          idx != -1);                  // has candidate split  
}

bool pExtSamp2VTTree::split_node( pExtSamp2VTNode* _node, pExtSamp2VTData* _data )
{
  // create left and right node
  pExtSamp2VTNode tmp1(nodes_.size(), K_);
  nodes_.push_back(tmp1);
  _node->left_ = &(nodes_.back());
  _node->left_->parent_ = _node;

  pExtSamp2VTNode tmp2(nodes_.size(), K_);
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

  // initialize node sovler
  _node->left_->sol_this_.set_data(_data, &(this->sub_ci_));
  _node->left_->sol_this_.update_internal(_node->left_->sample_idx_);
  _node->right_->sol_this_.set_data(_data, &(this->sub_ci_));
  _node->right_->sol_this_.update_internal(_node->right_->sample_idx_);

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

void pExtSamp2VTTree::calc_gain(pExtSamp2VTNode* _node, pExtSamp2VTData* _data)
{
  double gain;
  _node->sol_this_.calc_gain(gain);
  _node->split_.this_gain_ = gain;
}

void pExtSamp2VTTree::fit_node( pExtSamp2VTNode* _node, pExtSamp2VTData* _data )
{
  int nn = _node->sample_idx_.size();

#if 0
  CV_Assert(nn>0);
#endif
#if 1
  if (nn<=0) return;
#endif
  

  _node->sol_this_.calc_gamma( &(_node->fitvals_[0]) );

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

// Implementation of pExtSamp2VTLogitBoost::Param
pExtSamp2VTLogitBoost::Param::Param()
{
  T = 2;
  v = 0.1;
  J = 4;
  ns = 1;
  ratio_si_ = ratio_fi_ = ratio_ci_ = 0.6;
}
// Implementation of pExtSamp2VTLogitBoost
const double pExtSamp2VTLogitBoost::EPS_LOSS = 1e-14;
const double pExtSamp2VTLogitBoost::MAX_F = 100;
void pExtSamp2VTLogitBoost::train( MLData* _data )
{
  train_init(_data);

  for (int t = 0; t < param_.T; ++t) {
#ifdef OUTPUT
    os << "t = " << t << endl;
#endif // OUTPUT
    trees_[t].split(&logitdata_);
    trees_[t].fit(&logitdata_);

    update_F(t);
    update_p();
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

void pExtSamp2VTLogitBoost::predict( MLData* _data )
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

void pExtSamp2VTLogitBoost::predict( float* _sapmle, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  } // for k

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = 0; t < NumIter_; ++t) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    } // for k
  } // for t
}

void pExtSamp2VTLogitBoost::predict( MLData* _data, int _Tpre )
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

void pExtSamp2VTLogitBoost::predict( float* _sapmle, float* _score, int _Tpre )
{
  // IMPORTANT: caller should assure the validity of _Tpre

  // initialize 
  for (int k = 0; k < K_; ++k) 
    *(_score+k) = 0;

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = Tpre_beg_; t < _Tpre; ++t ) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    }
  }

}


int pExtSamp2VTLogitBoost::get_class_count()
{
  return K_;
}

int pExtSamp2VTLogitBoost::get_num_iter()
{
  return NumIter_;
}

//double pExtSamp2VTLogitBoost::get_train_loss()
//{
//  if (NumIter_<1) return DBL_MAX;
//  return L_iter_.at<double>(NumIter_-1);
//}

void pExtSamp2VTLogitBoost::train_init( MLData* _data )
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

  // gradient
  g_.create(N,1);
  g_ = 1.0; // OK as initial values

  // iteration for training
  NumIter_ = 0;

  // AOTOData
  logitdata_.data_cls_ = _data;
  logitdata_.p_ = &p_;
  logitdata_.L_ = &L_;
  logitdata_.g_ = &g_;

  // trees
  trees_.clear();
  trees_.resize(param_.T);
  for (int t = 0; t < param_.T; ++t) {
    trees_[t].param_.max_leaves_ = param_.J;
    trees_[t].param_.node_size_ = param_.ns;

    trees_[t].param_.ratio_ci_ = param_.ratio_ci_;
    trees_[t].param_.ratio_fi_ = param_.ratio_fi_;
    trees_[t].param_.ratio_si_ = param_.ratio_si_;
  }

  // gradient/delta
  abs_grad_.clear();
  abs_grad_.resize(param_.T);

  // for prediction
  Tpre_beg_ = 0;
}

void pExtSamp2VTLogitBoost::update_F(int t)
{
  int N = logitdata_.data_cls_->X.rows;
  double v = param_.v;
  vector<float> f(K_);
  for (int i = 0; i < N; ++i) {
    float *psample = logitdata_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);

    double* pF = F_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(pF+k) += (v*f[k]);
      // MAX cap
      if (*(pF+k) > MAX_F) *(pF+k) = MAX_F; // TODO: make the threshold a constant variable
    } // for k
  } // for i
}

void pExtSamp2VTLogitBoost::update_p()
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

//bool pExtSamp2VTLogitBoost::should_stop( int t )
//{
//  int N = F_.rows;
//  //double peps = exp(MIN_F-1); // min p <--> MIN_F
//  double delta = 0;
//    
//  for (int i = 0; i < N; ++i) {
//    double* ptr_pi = p_.ptr<double>(i);
//    int yi = int( logitdata_.data_cls_->Y.at<float>(i) );
//
//    for (int k = 0; k < K_; ++k) {
//      double pik = *(ptr_pi+k);
//      if (yi==k) delta += std::abs( 1-pik );
//      else       delta += std::abs( -pik );    
//    }
//  }
//  
//  abs_grad_[t] = delta;
//  if ( delta < (2*N*K_*1e-3) ) // TODO: make the threshold a constant variable
//    return true;
//  else
//    return false;
//}

void pExtSamp2VTLogitBoost::calc_loss( MLData* _data )
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

void pExtSamp2VTLogitBoost::calc_loss_iter( int t )
{
  double sum = 0;
  int N = L_.rows;
  for (int i = 0; i < N; ++i) 
    sum += L_.at<double>(i);

  L_iter_.at<double>(t) = sum;
}

bool pExtSamp2VTLogitBoost::should_stop( int t )
{
  double loss = L_iter_.at<double>(t);
  return ( (loss<EPS_LOSS) ? true : false );
}

void pExtSamp2VTLogitBoost::calc_grad( int t )
{
  int N = F_.rows;
  double delta = 0;
    
  for (int i = 0; i < N; ++i) {
    double* ptr_pi = p_.ptr<double>(i);
    int yi = int( logitdata_.data_cls_->Y.at<float>(i) );

    double ggg = 0.0;
    for (int k = 0; k < K_; ++k) {
      double pik = *(ptr_pi+k);
      if (yi==k) {
        delta += std::abs( 1-pik );
        ggg += std::abs( 1-pik );
      }
      else  {
        delta += std::abs( -pik ); 
        ggg += std::abs( -pik ); 
      }
    } // for k
    // update g_
    g_.at<double>(i) = ggg;
  }
  
  abs_grad_[t] = delta;
}



