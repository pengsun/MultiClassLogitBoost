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

// data shared by tree and booster
struct pVbExtSamp12SkimVTData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
  cv::Mat_<double> *gg_; // #samples * 1
};

// Solver
struct pVbExtSamp12SkimVTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pVbExtSamp12SkimVTSolver () {};
  pVbExtSamp12SkimVTSolver (pVbExtSamp12SkimVTData* _data, VecIdx* _ci);
  void set_data (pVbExtSamp12SkimVTData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pVbExtSamp12SkimVTData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pVbExtSamp12SkimVTSplit {
public:
  pVbExtSamp12SkimVTSplit ();
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
struct pVbExtSamp12SkimVTNode {
public:
  pVbExtSamp12SkimVTNode (int _K);
  pVbExtSamp12SkimVTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pVbExtSamp12SkimVTNode *parent_, *left_, *right_; //
  pVbExtSamp12SkimVTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds
  VecIdx sub_ci_; // subclass set
  pVbExtSamp12SkimVTSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pVbExtSamp12SkimVTNodeLess {
  bool operator () (const pVbExtSamp12SkimVTNode* n1, const pVbExtSamp12SkimVTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pVbExtSamp12SkimVTNode*, 
                            std::vector<pVbExtSamp12SkimVTNode*>, 
                            pVbExtSamp12SkimVTNodeLess>
        QuepVbExtSamp12SkimVTNode;

// Best Split Finder (helper class for parallel_reduce) 
class pVbExtSamp12SkimVTTree;
struct pVbExt12SkimVT_best_split_finder {
  pVbExt12SkimVT_best_split_finder (pVbExtSamp12SkimVTTree *_tree, 
    pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data); // customized data
  pVbExt12SkimVT_best_split_finder (const pVbExt12SkimVT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pVbExt12SkimVT_best_split_finder &rhs); // required

  pVbExtSamp12SkimVTTree *tree_;
  pVbExtSamp12SkimVTNode *node_;
  pVbExtSamp12SkimVTData *data_;
  pVbExtSamp12SkimVTSplit cb_split_;
};

// Tree
class pVbExtSamp12SkimVTTree {
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

public:
  void split( pVbExtSamp12SkimVTData* _data );
  void fit ( pVbExtSamp12SkimVTData* _data );

  pVbExtSamp12SkimVTNode* get_node (float* _sample);
  void get_is_leaf (VecInt& is_leaf);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample_samples (pVbExtSamp12SkimVTData* _data);
  void subsample_classes_for_node (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data);

  void clear ();
  void creat_root_node (pVbExtSamp12SkimVTData* _data);

  virtual bool find_best_candidate_split (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data);
  virtual bool find_best_split_num_var (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data, int _ivar,
                                        pVbExtSamp12SkimVTSplit &spl);

  void make_node_sorted_idx(pVbExtSamp12SkimVTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pVbExtSamp12SkimVTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pVbExtSamp12SkimVTSplit &cb_split);

  bool can_split_node (pVbExtSamp12SkimVTNode* _node);
  bool split_node (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data);
  void calc_gain (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data);
  virtual void fit_node (pVbExtSamp12SkimVTNode* _node, pVbExtSamp12SkimVTData* _data);

protected:
  std::list<pVbExtSamp12SkimVTNode> nodes_; // all nodes
  QuepVbExtSamp12SkimVTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes
};

// vector of pVbExtSamp12SkimVTTree
typedef std::vector<pVbExtSamp12SkimVTTree> VecpVbExtSamp12SkimVTTree;

// Boost
class pVbExtSamp12SkimVTLogitBoost {
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

protected:
  void train_init (MLData* _data);
  void update_F(int t);
  void update_p();
  void update_gg ();

  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  bool should_stop (int t);
  void calc_grad( int t );

public:
  std::vector<double> abs_grad_; // total gradient. indicator for stopping. 
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
protected:
  int K_; // class count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  cv::Mat_<double> gg_; // gradient. #samples * #classes

  int NumIter_; // actual iteration number
  pVbExtSamp12SkimVTData logitdata_;
  VecpVbExtSamp12SkimVTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
