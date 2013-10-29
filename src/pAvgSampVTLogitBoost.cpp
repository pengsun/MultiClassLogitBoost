#include "pAvgSampVTLogitBoost.hpp"
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
    IdxMatGreater (MatDbl *_v) { v_ = _v;};

    bool operator () (int i1, int i2) {
      return ( v_->at<double>(i1) > v_->at<double>(i2) );
    };
    MatDbl *v_;
  };

  void release_VecDbl (VecDbl &in) {
    VecDbl tmp;
    tmp.swap(in);
  }

  void release_VecIdx (VecIdx &in) {
    VecIdx tmp;
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

  void weight_trim_ratio2 (cv::Mat_<double>& w, double ratio, VecIdx &ind, int Nmin) {
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
    const double eps = 1e-11;
    for (int i = 0; i < N; ++i) {
      int ii = wind[i];
      ind.push_back(ii);
      //
      if ( ind.size()>=Nmin ) break;
      //
      sum += (w.at<double>(ii)+eps)/(Z+eps); // prevent numeric problem
      if (ratio>1.0) continue; // never >= ratio...
      if ( sum>ratio ) break;
    } // for i
  }



  void calc_grad_1norm_samp (MatDbl& gg, MatDbl& out) {
    int nrow = gg.rows;
    out.create(nrow,1);

    int ncol = gg.cols;
    for (int i = 0; i < nrow; ++i) {
      double s = 0.0;
      for (int j = 0; j < ncol; ++j) {
        s += std::abs( gg.at<double>(i,j) );
      } // for j
      
      // update out
      out.at<double>(i) = s;
    } // for i

  } // calc_grad_1norm_samp

  void calc_grad_1norm_class (MatDbl& gg, MatDbl& out) {
    int ncol = gg.cols;
    out.create(ncol,1);

    int nrow = gg.rows;
    for (int j = 0; j < ncol; ++j) {
      double s = 0.0;
      for (int i = 0; i < nrow; ++i) {
        s += std::abs( gg.at<double>(i,j) );
      } // for i

      // update out
      out.at<double>(j) = std::abs(s);
    } // for j
  }


  void subsample_rows(MatDbl& in, const VecIdx& idx, MatDbl& out) {
    int K = in.cols;
    out.create(idx.size(),K);

    for (int i = 0; i < idx.size(); ++i) {
      int ii = idx[i];

      double *pfrom_beg = in.ptr<double>(i);
      double *pto_beg = out.ptr<double>(i);
      std::copy(pfrom_beg,pfrom_beg+K, pto_beg);
    }
  }
}

// Implementation of pAvgSampVTSplit
pAvgSampVTSplit::pAvgSampVTSplit()
{
  reset();
}

void pAvgSampVTSplit::reset()
{
  var_idx_ = -1;
  threshold_ = FLT_MAX;
  subset_.reset();

  this_gain_ = -1;
  expected_gain_ = -1;
  left_node_gain_ = right_node_gain_ = -1;
}


// Implementation of pAvgSampVTSolver
const double pAvgSampVTSolver::MAXGAMMA = 5.0;
pAvgSampVTSolver::pAvgSampVTSolver( pAvgSampVTData* _data, VecIdx* _ci)
{ 
  set_data(_data, _ci);
}

void pAvgSampVTSolver::set_data( pAvgSampVTData* _data, VecIdx* _ci)
{ 
  data_ = _data;
  ci_ = _ci;

  int KK = _ci->size();
  mg_.assign(KK, 0.0);
  h_.assign(KK, 0.0);
}

void pAvgSampVTSolver::update_internal( VecIdx& vidx )
{
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    update_internal_incre(idx);
  } // for it
}

void pAvgSampVTSolver::update_internal_incre( int idx )
{
  
  double* ptr_gg = data_->mgg_->ptr<double>(idx);;

  // mg and h
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    int k = this->ci_->at(kk);
    double ggik = *(ptr_gg + k);

    mg_[kk] += ggik;
    h_[kk]  += 1;
  }
}

void pAvgSampVTSolver::update_internal_decre( int idx )
{
  double* ptr_gg = data_->mgg_->ptr<double>(idx);;

  // mg and h
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    int k = this->ci_->at(kk);
    double ggik = *(ptr_gg + k);

    mg_[kk] -= ggik;
    h_[kk]  -= 1;
  }
}

void pAvgSampVTSolver::calc_gamma( double *gamma)
{
  int KK = mg_.size();
  for (int kk = 0; kk < KK; ++kk) {
    double smg = mg_[kk];
    double sh  = h_[kk];
    //if (sh <= 0) cv::error("pAvgSampVTSolver::calc_gamma: Invalid Hessian.");
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

void pAvgSampVTSolver::calc_gain( double& gain )
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


// Implementation of pVbExtSamp12VTNode
pAvgSampVTNode::pAvgSampVTNode(int _K)
{
  id_ = 0;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);

  gain_ = allsample_gain_ = 0.0;
}

pAvgSampVTNode::pAvgSampVTNode( int _id, int _K )
{
  id_ = _id;
  parent_ = left_ = right_ = 0;
  
  fitvals_.assign(_K, 0);

  gain_ = allsample_gain_ = 0.0;
}

int pAvgSampVTNode::calc_dir( float* _psample )
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

// Implementation of pAvgSampVTBestSplitFinder (helper class)
pAvgSampVTBestSplitFinder::pAvgSampVTBestSplitFinder(pAvgSampVTTree *_tree, 
  pAvgSampVTNode *_node, pAvgSampVTData *_data)
{
  this->tree_ = _tree;
  this->node_ = _node;
  this->data_ = _data;

  this->cb_split_.reset();
  this->cb_split_.this_gain_ = _node->split_.this_gain_;
}

pAvgSampVTBestSplitFinder::pAvgSampVTBestSplitFinder (const pAvgSampVTBestSplitFinder &f, cv::Split)
{
  this->tree_ = f.tree_;
  this->node_ = f.node_;
  this->data_ = f.data_;
  this->cb_split_ = f.cb_split_;
}

void pAvgSampVTBestSplitFinder::operator() (const cv::BlockedRange &r)
{

  // for each variable, find the best split
  for (int ii = r.begin(); ii != r.end(); ++ii) {
    int vi = this->tree_->sub_fi_[ii];

    pAvgSampVTSplit the_split;
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

void pAvgSampVTBestSplitFinder::join (pAvgSampVTBestSplitFinder &rhs)
{
  if ( rhs.cb_split_.expected_gain_ > (this->cb_split_.expected_gain_) ) {
    (this->cb_split_) = (rhs.cb_split_);
  }
}

// Implementation of pAvgSampVTInstSampler (helper class)
pAvgSampVTInstSampler::pAvgSampVTInstSampler()
{
  Tinner_ = 0;
  wsum_ = 0.0;
}

void pAvgSampVTInstSampler::reset( MatDbl _w, int _n )
{
  Tinner_ = 0;
  w_ = _w;
  n_ = _n;
  wsum_ = 0.0;

  // sorting the weights
  int N = w_.rows;
  wind_.resize(N);
  for (int i = 0; i < N; ++i) wind_[i] = i;
  std::sort (wind_.begin(),wind_.end(), IdxMatGreater(&w_));
}

int pAvgSampVTInstSampler::get_Tinner()
{
  return Tinner_;
}

void pAvgSampVTInstSampler::sample( VecIdx& idx )
{
  //
  idx.clear();

  // subsampling
  int n_beg = Tinner_*n_;
  for (int i = n_beg; i < (n_beg + n_); ++i) {
    if (i >= wind_.size()) break;

    int ii = wind_[i];
    idx.push_back(ii);
  }

  // update
  ++Tinner_;
}

void pAvgSampVTInstSampler::sample_uniform( VecIdx &idx )
{
  int N = w_.rows;
  double ratio = double(n_)/double(N);
  uniform_subsample_ratio (N, ratio, idx);
}

// Implementation of pAvgSampVTTree::Param
pAvgSampVTTree::Param::Param()
{
  max_leaves_ = 2;
  node_size_ = 5;
  ratio_si_ = ratio_fi_ = 0.6;
  ratio_ci_ = 0.8;

}

// Implementation of pAvgSampVTTree
void pAvgSampVTTree::split( pAvgSampVTData* _data )
{
  // clear
  clear();
  K_ = _data->mgg_->cols;

  // susampling instances already done.

  // root node
  creat_root_node(_data);
  candidate_nodes_.push(&nodes_.front());
  pAvgSampVTNode* root = candidate_nodes_.top(); 
  find_best_candidate_split(root, _data);
  int nleaves = 1;

  // split recursively
  while ( nleaves < param_.max_leaves_ &&
          !candidate_nodes_.empty() )
  {
    pAvgSampVTNode* cur_node = candidate_nodes_.top(); // the most prior node
    candidate_nodes_.pop();
    --nleaves;

    if (!can_split_node(cur_node)) { // can not split, make it a leaf
      ++nleaves;
      continue;
    }

    split_node(cur_node,_data);
    // release memory.
    // no longer used in later splitting
    release_VecIdx(cur_node->sample_idx_);
    release_VecIdx(cur_node->allsample_idx_);
    release_VecIdx(cur_node->sub_ci_);
    release_VecDbl(cur_node->sol_this_.mg_);
    release_VecDbl(cur_node->sol_this_.h_);

    // find best split for the two newly created nodes
    find_best_candidate_split(cur_node->left_, _data);
    candidate_nodes_.push(cur_node->left_);
    ++nleaves;

    find_best_candidate_split(cur_node->right_, _data);
    candidate_nodes_.push(cur_node->right_);
    ++nleaves;
  }
}

void pAvgSampVTTree::fit( pAvgSampVTData* _data )
{
  // fitting node data for each leaf
  std::list<pAvgSampVTNode>::iterator it;
  for (it = nodes_.begin(); it != nodes_.end(); ++it) {
    pAvgSampVTNode* nd = &(*it);

    if (nd->left_!=0) { // not a leaf
      continue;
    } 

    // node values
    fit_node(nd,_data);

    // all samples node gain
    calc_allsample_node_gain(nd,_data);
    // update
    leaf_allsample_gain_.push_back(nd->allsample_gain_);

    // node gain
    nd->gain_ = nd->split_.this_gain_;
    // update
    leaf_gain_.push_back(nd->gain_);

    // release memory.
    // no longer used in later splitting
    release_VecIdx(nd->sample_idx_);
    release_VecIdx(nd->allsample_idx_);
    release_VecIdx(nd->sub_ci_);
    release_VecDbl( nd->sol_this_.mg_ );
    release_VecDbl( nd->sol_this_.h_ );
  }
}


pAvgSampVTNode* pAvgSampVTTree::get_node( float* _sample)
{
  pAvgSampVTNode* cur_node = &(nodes_.front());
  while (true) {
    if (cur_node->left_==0) break; // leaf reached 

    int dir = cur_node->calc_dir(_sample);
    pAvgSampVTNode* next = (dir==-1) ? (cur_node->left_) : (cur_node->right_);
    cur_node = next;
  }
  return cur_node;
}


void pAvgSampVTTree::get_is_leaf( VecInt& is_leaf )
{
  is_leaf.clear();
  is_leaf.resize(nodes_.size());

  std::list<pAvgSampVTNode>::iterator it = nodes_.begin();
  for (int i = 0; it!=nodes_.end(); ++it, ++i) {
    if (it->left_==0 && it->right_==0)
      is_leaf[i] = 1;
    else
      is_leaf[i] = 0;
  } // for i
}

void pAvgSampVTTree::predict( MLData* _data )
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

void pAvgSampVTTree::predict( float* _sample, float* _score )
{
  // initialize all the *K* classes
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update all the K classes...
  pAvgSampVTNode* nd = get_node(_sample);
  for (int k = 0; k < K_; ++k) {
    float val = static_cast<float>( nd->fitvals_[k] );
    *(_score + k) = val;
  }

}


void pAvgSampVTTree::subsample_classes_for_node( pAvgSampVTNode* _node, pAvgSampVTData* _data )
{
  /// subsample_samples classes

  // class wise, use the examples that the node holds
  MatDbl tmp;
  subsample_rows(*_data->mgg_, _node->sample_idx_, tmp);
  cv::Mat_<double> g_class; 
  calc_grad_1norm_class(tmp, g_class); // after: K * 1

  // minimum #classes
  int MIN_CLASS = int( double(K_)*double(param_.ratio_ci_) );

  VecIdx ci_wt;
  weight_trim_ratio2(g_class, this->param_.weight_ratio_ci_, ci_wt, MIN_CLASS);
  // set it
  _node->sub_ci_ = ci_wt;
}

void pAvgSampVTTree::clear()
{
  nodes_.clear();
  candidate_nodes_.~priority_queue();

  node_cc_.clear();
  node_sc_.clear();
  node_all_sc_.clear();
}

void pAvgSampVTTree::creat_root_node( pAvgSampVTData* _data )
{
  nodes_.push_back(pAvgSampVTNode(0,K_));
  pAvgSampVTNode* root = &(nodes_.back());

  // samples in node
  int NN = this->sub_si_.size();
  root->sample_idx_.resize(NN);
  for (int ii = 0; ii < NN; ++ii) {
    int ind = sub_si_[ii];
    root->sample_idx_[ii] = ind;
  }

  // all samples in node
  int NALL = _data->mgg_->rows;
  root->allsample_idx_.resize(NALL);
  for (int j = 0; j < NALL; ++j) 
    root->allsample_idx_[j] = j;

  // subsample_samples classes for the current node (node level)
  subsample_classes_for_node(root, _data);

  // updaate node class count
  this->node_cc_.push_back( root->sub_ci_.size() );
  // update node sample count
  this->node_sc_.push_back( root->sample_idx_.size() );
  //
  this->node_all_sc_.push_back(root->allsample_idx_.size());

  // initialize solver
  root->sol_this_.set_data(_data, &(root->sub_ci_));
  root->sol_this_.update_internal(root->sample_idx_);

  // loss
  this->calc_node_gain(root, _data);
}

bool pAvgSampVTTree::find_best_candidate_split( pAvgSampVTNode* _node, pAvgSampVTData* _data )
{
  // subsample the features for the current node (node level)
  int NF = _data->data_cls_->X.cols;
  uniform_subsample_ratio(NF,this->param_.ratio_fi_, this->sub_fi_);

  // the range (beginning/ending variable)
  int nsubvar = this->sub_fi_.size();
  cv::BlockedRange br(0,nsubvar,1);

  // do the search in parallel
  pAvgSampVTBestSplitFinder bsf(this,_node,_data);
  cv::parallel_reduce(br, bsf);

  // update node's split
  _node->split_ = bsf.cb_split_;
  return true; // TODO: Check if this is reasonable

}

bool pAvgSampVTTree::find_best_split_num_var( 
  pAvgSampVTNode* _node, pAvgSampVTData* _data, int _ivar, pAvgSampVTSplit &cb_split)
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
  pAvgSampVTSolver sol_left(_data, _node->sol_this_.ci_), 
                      sol_right = _node->sol_this_;

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

void pAvgSampVTTree::make_node_sorted_idx( pAvgSampVTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node )
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

bool pAvgSampVTTree::set_best_split_num_var( 
  pAvgSampVTNode* _node, MLData* _data, int _ivar, 
  VecIdx& node_sample_si, 
  int best_i, double best_gain, double best_gain_left, double best_gain_right,
  pAvgSampVTSplit &cb_split)
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

bool pAvgSampVTTree::can_split_node( pAvgSampVTNode* _node )
{
  bool flag = true;
  int nn = _node->sample_idx_.size();
  int idx = _node->split_.var_idx_;
  return (nn > param_.node_size_    && // large enough node size
          idx != -1);                  // has candidate split  
}

bool pAvgSampVTTree::split_node( pAvgSampVTNode* _node, pAvgSampVTData* _data )
{
  // create left and right node
  pAvgSampVTNode tmp1(nodes_.size(), K_);
  nodes_.push_back(tmp1);
  _node->left_ = &(nodes_.back());
  _node->left_->parent_ = _node;

  pAvgSampVTNode tmp2(nodes_.size(), K_);
  nodes_.push_back(tmp2);
  _node->right_ = &(nodes_.back());
  _node->right_->parent_ = _node;

  // send each sample (after sampling) to left/right node
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

  // send each sample (all) to left/right node
  int nall = _node->allsample_idx_.size();
  for (int i = 0; i < nall; ++i) {
    int idx = _node->allsample_idx_[i];

    float* p = (float*)data_cls->X.ptr(idx);
    int dir = _node->calc_dir(p);
    if (dir == -1) 
      _node->left_->allsample_idx_.push_back(idx);
    else 
      _node->right_->allsample_idx_.push_back(idx);
  }

  // update current node's status: leaf -> internal node
  

  // subsample_samples classes for the two newly created nodes
  subsample_classes_for_node(_node->left_, _data);
  this->node_cc_.push_back( _node->left_->sub_ci_.size() );
  this->node_sc_.push_back( _node->left_->sample_idx_.size() );
  this->node_all_sc_.push_back(_node->left_->allsample_idx_.size());

  //
  subsample_classes_for_node(_node->right_, _data);
  this->node_cc_.push_back( _node->right_->sub_ci_.size() );
  this->node_sc_.push_back( _node->right_->sample_idx_.size() );
  this->node_all_sc_.push_back(_node->right_->allsample_idx_.size());


  // initialize node sovler
  _node->left_->sol_this_.set_data(_data, &(_node->left_->sub_ci_));
  _node->left_->sol_this_.update_internal(_node->left_->sample_idx_);
  _node->right_->sol_this_.set_data(_data, &(_node->right_->sub_ci_));
  _node->right_->sol_this_.update_internal(_node->right_->sample_idx_);

  // initialize the node gain
  this->calc_node_gain(_node->left_, _data);
  this->calc_node_gain(_node->right_, _data);

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

void pAvgSampVTTree::fit_node( pAvgSampVTNode* _node, pAvgSampVTData* _data )
{
  int nn = _node->sample_idx_.size();
  if (nn<=0) return;

#if 1
  // Use all the classes to update node values
  VecIdx ci(this->K_);
  for (int k = 0; k < this->K_; ++k)
    ci[k] = k;
  pAvgSampVTSolver sol(_data, &ci);
  sol.update_internal(_node->sample_idx_);
  sol.calc_gamma( &(_node->fitvals_[0]) );
#endif

#if 0
  _node->sol_this_.calc_gamma( &(_node->fitvals_[0]) );
#endif

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

void pAvgSampVTTree::calc_node_gain(pAvgSampVTNode* _node, pAvgSampVTData* _data)
{
  double gain;
  _node->sol_this_.calc_gain(gain);
  _node->split_.this_gain_ = gain;
}

void pAvgSampVTTree::calc_allsample_node_gain( pAvgSampVTNode* _node, pAvgSampVTData* _data )
{
  // Use all the classes & all the examples to calc the gain
  VecIdx ci(this->K_);
  for (int k = 0; k < this->K_; ++k)
    ci[k] = k;
  pAvgSampVTSolver sol(_data, &ci);
  sol.update_internal(_node->allsample_idx_);

  sol.calc_gain( _node->allsample_gain_ );
}

// Implementation of pAvgSampVTLogitBoost::Param
pAvgSampVTLogitBoost::Param::Param()
{
  T = 2;
  v = 0.1;
  J = 4;
  ns = 1;
  ratio_si_ = ratio_fi_ = ratio_ci_ = 0.6;
  weight_ratio_ci_ = weight_ratio_si_ = 0.6;
  Tdot = 3;

}

// Implementation of pAvgSampVTLogitBoost
const double pAvgSampVTLogitBoost::EPS_LOSS = 1e-6;
const double pAvgSampVTLogitBoost::MAX_F = 100;
void pAvgSampVTLogitBoost::train( MLData* _data )
{
  train_init(_data);

  bool flag = false;
  int t = 0;
  while (true) {
    // inner iteration: 
    // approximate gradient by averaging several trees
    for (int tt = 0; tt < param_.Tdot; ++tt) {
      if (t>=param_.T) {
        flag = true;
        break;
      }

      // subsample instances
      subsample_inst( trees_[t].sub_si_ );

      // fit a tree using just the subset
      trees_[t].split(&data_);
      trees_[t].fit(&data_);

      // update the gradient (residual)
      calc_mgg_residual(t);

      calc_F(t);
      calc_abs_grad_class(t);
      calc_grad(t);

      calc_loss(_data);
      calc_loss_iter(t);
      calc_loss_class(_data,t);

      // update #iterations
      NumIter_ = t + 1;

      // if break
      if ( should_stop(t) ) {
        flag = true;
        break;
      }

      // 
      ++t;
    } // for tt

    if (flag) break;

    update_p_using_F();
    update_mgg_using_p();

  } // while

}

void pAvgSampVTLogitBoost::predict( MLData* _data )
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

void pAvgSampVTLogitBoost::predict( float* _sapmle, float* _score )
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

void pAvgSampVTLogitBoost::predict( MLData* _data, int _Tpre )
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

void pAvgSampVTLogitBoost::predict( float* _sapmle, float* _score, int _Tpre )
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


int pAvgSampVTLogitBoost::get_class_count()
{
  return K_;
}

int pAvgSampVTLogitBoost::get_num_iter()
{
  return NumIter_;
}

void pAvgSampVTLogitBoost::get_nr( VecIdx& nr_wts, VecIdx& nr_wtc )
{
  nr_wts.resize(this->NumIter_);
  nr_wtc.resize(this->NumIter_);

  for (int i = 0; i < NumIter_; ++i) {
    nr_wts[i] = trees_[i].node_sc_[0];
    nr_wtc[i] = trees_[i].node_cc_[0];
  }

}

void pAvgSampVTLogitBoost::get_cc( int itree, VecInt& node_cc )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  node_cc = trees_[itree].node_cc_;
}


void pAvgSampVTLogitBoost::get_sc( int itree, VecInt& node_sc )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  node_sc = trees_[itree].node_sc_;
}

//double pAvgSampVTLogitBoost::get_train_loss()
//{
//  if (NumIter_<1) return DBL_MAX;
//  return L_iter_.at<double>(NumIter_-1);
//}

void pAvgSampVTLogitBoost::get_is_leaf( int itree, VecInt& is_leaf )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  trees_[itree].get_is_leaf(is_leaf);
}


void pAvgSampVTLogitBoost::get_all_sc( int itree, VecInt& node_all_sc )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  node_all_sc = trees_[itree].node_all_sc_;
}

void pAvgSampVTLogitBoost::get_gain( int itree, VecDbl& leaf_gain )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  leaf_gain = trees_[itree].leaf_gain_;
}

void pAvgSampVTLogitBoost::get_allsample_gain( int itree, VecDbl& leaf_allsample_gain )
{
  if (NumIter_==0) return;
  if (itree<0) itree = 0;
  if ( (itree+1) >= NumIter_ ) itree = (NumIter_-1);

  leaf_allsample_gain = trees_[itree].leaf_allsample_gain_;
}

void pAvgSampVTLogitBoost::train_init( MLData* _data )
{
  // class count
  K_ = _data->get_class_count();

  // F, p
  int N = _data->X.rows;
  F_.create(N,K_); 
  F_ = 1;
  p_.create(N,K_); 
  update_p_using_F();

  // Loss
  L_.create(N,1);
  calc_loss(_data);
  L_iter_.create(param_.T,1);

  // class loss
  loss_class_.create(param_.T, K_);
  loss_class_ = 0.0;

  // iteration for training
  NumIter_ = 0;

  // AOTOData
  data_.data_cls_ = _data;

  // gradient & set AOTOData
  mgg_.create(N,K_);
  update_mgg_using_p();
  data_.mgg_ = &mgg_;

  // trees
  trees_.clear();
  trees_.resize(param_.T);
  for (int t = 0; t < param_.T; ++t) {
    trees_[t].param_.max_leaves_ = param_.J;
    trees_[t].param_.node_size_ = param_.ns;

    trees_[t].param_.ratio_ci_ = param_.ratio_ci_;
    trees_[t].param_.ratio_fi_ = param_.ratio_fi_;
    trees_[t].param_.ratio_si_ = param_.ratio_si_;
    trees_[t].param_.weight_ratio_ci_ = param_.weight_ratio_ci_;
    trees_[t].param_.weight_ratio_si_ = param_.weight_ratio_si_;
  }

  // gradient/delta
  abs_grad_.clear();
  abs_grad_.resize(param_.T);

  // class gradient
  abs_grad_class_.create(param_.T, K_);
  abs_grad_class_ = 0.0;

  // for prediction
  Tpre_beg_ = 0;
}

void pAvgSampVTLogitBoost::calc_F(int t)
{
  int N = data_.data_cls_->X.rows;
  double v = param_.v;
  vector<float> f(K_);
  for (int i = 0; i < N; ++i) {
    float *psample = data_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);

    double* pF = F_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(pF+k) += (v*f[k]);
      // MAX cap
      if (*(pF+k) > MAX_F) *(pF+k) = MAX_F; // TODO: make the threshold a constant variable
    } // for k
  } // for i
}


void pAvgSampVTLogitBoost::calc_mgg_residual( int t )
{
  int N = data_.data_cls_->X.rows;

  vector<float> f(K_);
  for (int i = 0; i < N; ++i) {
    float *psample = data_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);

    double* ptr_mgg = mgg_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(ptr_mgg+k) -= (0.025*f[k]);
    } // for k
  } // for i
}

void pAvgSampVTLogitBoost::update_p_using_F()
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

void pAvgSampVTLogitBoost::update_mgg_using_p()
{
  int N = F_.rows;
  double delta = 0;

  for (int i = 0; i < N; ++i) {
    double* ptr_pi = p_.ptr<double>(i);
    int yi = int( data_.data_cls_->Y.at<float>(i) );

    for (int k = 0; k < K_; ++k) {
      double pik = *(ptr_pi+k);

      if (yi==k) 
        mgg_.at<double>(i,k) = (1-pik);
      else  
        mgg_.at<double>(i,k) = (-pik);
    } // for k
  } // for i
}

void pAvgSampVTLogitBoost::calc_abs_grad_class(int t)
{
  int N = F_.rows;

  for (int k = 0; k < K_; ++k) {
    double sum = 0;
    
    for (int i = 0; i < N; ++i) {
      sum += abs( mgg_.at<double>(i,k) );
    }
    
    abs_grad_class_.at<double>(t,k) = sum;
  }
}


void pAvgSampVTLogitBoost::subsample_inst(VecIdx& ind)
{

  // gradient's norm as weights
  cv::Mat_<double> w;
  calc_grad_1norm_samp( *data_.mgg_, w); // sample wise, after: N * 1

  // #samples to be sampled
  int N = data_.data_cls_->X.rows;
  int n = int( double(N)*double(this->param_.ratio_si_) );

  // 
  sampler_.reset(w, n);
  
  // do the subsampling
  sampler_.sample(ind);
  //sampler_.sample_uniform(ind);
}

void pAvgSampVTLogitBoost::calc_loss( MLData* _data )
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

void pAvgSampVTLogitBoost::calc_loss_iter( int t )
{
  double sum = 0;
  int N = L_.rows;
  for (int i = 0; i < N; ++i) 
    sum += L_.at<double>(i);

  L_iter_.at<double>(t) = sum;
}

void pAvgSampVTLogitBoost::calc_loss_class(MLData* _data,  int t )
{
  const double PMIN = 0.0001;
  int N = _data->X.rows;
  for (int i = 0; i < N; ++i) {
    int yi = int( _data->Y.at<float>(i) );
    double* ptr = p_.ptr<double>(i);
    double pik = *(ptr + yi);

    if (pik<PMIN) pik = PMIN;
    loss_class_.at<double>(t,yi) += (-log(pik));
  }
}

bool pAvgSampVTLogitBoost::should_stop( int t )
{
  // stop if too small #classes subsampled
  if ( ! (trees_[t].node_cc_.empty()) ) {
    if (trees_[t].node_cc_[0] == 1) // only one class selected for root node
      return true;
  }

  // stop if loss is small enough
  double loss = L_iter_.at<double>(t);
  return ( (loss<EPS_LOSS) ? true : false );
}

void pAvgSampVTLogitBoost::calc_grad( int t )
{
  int N = F_.rows;
  double delta = 0;

  for (int i = 0; i < N; ++i) {
    double* ptr_pi = p_.ptr<double>(i);
    int yi = int( data_.data_cls_->Y.at<float>(i) );

    for (int k = 0; k < K_; ++k) {
      double pik = *(ptr_pi+k);
      if (yi==k) {
        delta += std::abs( 1-pik );
      }
      else  {
        delta += std::abs(pik);
      }
    } // for k
  }

  abs_grad_[t] = delta;
}
