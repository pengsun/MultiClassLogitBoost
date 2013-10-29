#pragma once

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>
#include <opencv2/core/internal.hpp>

// Vector for index
typedef std::vector<int> VecIdx;
typedef std::vector<int> VecInt;
typedef std::vector<double> VecDbl;
typedef cv::Mat_<double> MatDbl;

// data shared by tree and booster
struct pAvgSampVTData {
  MLData *data_cls_;
  MatDbl *mgg_; // #samples * #class
};

// Solver
struct pAvgSampVTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pAvgSampVTSolver () {};
  pAvgSampVTSolver (pAvgSampVTData* _data, VecIdx* _ci);
  void set_data (pAvgSampVTData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pAvgSampVTData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pAvgSampVTSplit {
public:
  pAvgSampVTSplit ();
  void reset ();

  int var_idx_; // variable for split
  VAR_TYPE var_type_; // variable type

  float threshold_; // for numeric variable 
  std::bitset<MLData::MAX_VAR_CAT> subset_; // mask for category variable, maximum 64 #category 

  double this_gain_;
  double expected_gain_;
  double left_node_gain_, right_node_gain_;
};

// AOSO Node. Vector value
struct pAvgSampVTNode {
public:
  pAvgSampVTNode (int _K);
  pAvgSampVTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pAvgSampVTNode *parent_, *left_, *right_; //
  pAvgSampVTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds
  VecIdx sub_ci_; // subclass set
  pAvgSampVTSolver sol_this_; // for all the examples this node holds

  VecIdx allsample_idx_;
  double gain_;
  double allsample_gain_;
};

// Node Comparator: the less the expected gain, the less the node
struct pAvgSampVTNodeLess {
  bool operator () (const pAvgSampVTNode* n1, const pAvgSampVTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pAvgSampVTNode*, 
                            std::vector<pAvgSampVTNode*>, 
                            pAvgSampVTNodeLess>
        QuepAvgSampVTNode;

// Best Split Finder (helper class for parallel_reduce) 
class pAvgSampVTTree;
struct pAvgSampVTBestSplitFinder {
  pAvgSampVTBestSplitFinder (pAvgSampVTTree *_tree, 
    pAvgSampVTNode* _node, pAvgSampVTData* _data); // customized data
  pAvgSampVTBestSplitFinder (const pAvgSampVTBestSplitFinder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pAvgSampVTBestSplitFinder &rhs); // required

  pAvgSampVTTree *tree_;
  pAvgSampVTNode *node_;
  pAvgSampVTData *data_;
  pAvgSampVTSplit cb_split_;
};

// Sampler (helper class for instance sampling)
struct pAvgSampVTInstSampler {
  pAvgSampVTInstSampler ();

  void reset (MatDbl _w, int _n);
  int get_Tinner ();
  void sample (VecIdx &idx);
  void sample_uniform (VecIdx &idx);

private:
  MatDbl w_; // weights
  int Tinner_; // inner iterations, i.e., number of being called
  int n_; // number of instances to be sampled

  double wsum_; // weighted sum

  VecIdx wind_; // weights index after sorting
};

// Tree
class pAvgSampVTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double ratio_si_, ratio_fi_, ratio_ci_;
    double weight_ratio_si_, weight_ratio_ci_; 
    Param ();
  };
  Param param_;

public:
  // TODO: Make sure of Thread Safety!
  VecIdx sub_si_, sub_fi_, sub_ci_; // sample/feature/class index after subsample_samples 

public:
  VecInt node_cc_; // class count for each node   
  VecInt node_sc_; // sample count for each node
  VecInt node_all_sc_; // all sample count for each node
  VecDbl leaf_gain_;
  VecDbl leaf_allsample_gain_;

public:
  void split( pAvgSampVTData* _data );
  void fit ( pAvgSampVTData* _data );

  pAvgSampVTNode* get_node (float* _sample);
  void get_is_leaf (VecInt& is_leaf);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample_classes_for_node (pAvgSampVTNode* _node, pAvgSampVTData* _data);

  void clear ();
  void creat_root_node (pAvgSampVTData* _data);

  virtual bool find_best_candidate_split (pAvgSampVTNode* _node, pAvgSampVTData* _data);
  virtual bool find_best_split_num_var (pAvgSampVTNode* _node, pAvgSampVTData* _data, int _ivar,
                                        pAvgSampVTSplit &spl);

  void make_node_sorted_idx(pAvgSampVTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pAvgSampVTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pAvgSampVTSplit &cb_split);

  bool can_split_node (pAvgSampVTNode* _node);
  bool split_node (pAvgSampVTNode* _node, pAvgSampVTData* _data);
  virtual void fit_node (pAvgSampVTNode* _node, pAvgSampVTData* _data);

  void calc_node_gain (pAvgSampVTNode* _node, pAvgSampVTData* _data);
  void calc_allsample_node_gain (pAvgSampVTNode* _node, pAvgSampVTData* _data);

protected:
  std::list<pAvgSampVTNode> nodes_; // all nodes
  QuepAvgSampVTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  int K_; // #classes

};

// vector of pAvgSampVTTree
typedef std::vector<pAvgSampVTTree> VecpAvgSampVTTree;

// Boost
class pAvgSampVTLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAX_F;

public:
  struct Param {
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
    double ratio_si_, ratio_fi_, ratio_ci_; // ratios for subsampling
    double weight_ratio_si_, weight_ratio_ci_; // weight ratios for subsampling
    int Tdot;
    Param ();
  };
  Param param_;
  
public:
  void train (MLData* _data);

  void predict (MLData* _data);
  virtual void predict (float* _sapmle, float* _score);
  void predict (MLData* _data, int _Tpre);
  virtual void predict (float* _sapmle, float* _score, int _Tpre);

  int get_class_count ();
  int get_num_iter ();
  //double get_train_loss ();
  void get_nr (VecIdx& nr_wts, VecIdx& nr_wtc);
  void get_cc (int itree, VecInt& node_cc);
  void get_sc (int itree, VecInt& node_sc);
  void get_is_leaf (int itree, VecInt& is_leaf);

  void get_all_sc (int itree, VecInt& node_all_sc);
  void get_gain (int itree, VecDbl& leaf_gain);
  void get_allsample_gain (int itree, VecDbl& leaf_allsample_gain);

public:
  std::vector<double> abs_grad_; // total gradient. indicator for stopping. 
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
  cv::Mat_<double> abs_grad_class_; // #iteration * #classes
  cv::Mat_<double> loss_class_;

protected:
  void train_init (MLData* _data);

  void subsample_inst (VecIdx& ind);

  void calc_F(int t);
  void calc_mgg_residual(int t);

  void calc_abs_grad_class (int t);
  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  void calc_loss_class (MLData* _data, int t);
  void calc_grad( int t );

  void update_p_using_F();
  void update_mgg_using_p ();

  bool should_stop (int t);

protected:
  int K_; // class count
  MatDbl L_; // Loss. #samples
  MatDbl L_iter_; // Loss. #iteration
  MatDbl mgg_; // gradient. #samples * #classes

  int NumIter_; // actual iteration number
  pAvgSampVTData data_;
  VecpAvgSampVTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  MatDbl Fpre_; // Score for test data. #samples * #class

  pAvgSampVTInstSampler sampler_;
};
