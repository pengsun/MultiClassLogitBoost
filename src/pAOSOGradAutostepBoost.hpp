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
struct pAOSOAutostepGradData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
  cv::Mat_<double> *gg_; // #samples * 1
};

// Solver
struct pAOSOAutostepGradSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pAOSOAutostepGradSolver () {};
  pAOSOAutostepGradSolver (pAOSOAutostepGradData* _data, VecIdx* _ci);
  void set_data (pAOSOAutostepGradData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;

  pAOSOAutostepGradData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pAOSOAutostepGradSplit {
public:
  pAOSOAutostepGradSplit ();
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
struct pAOSOAutostepGradNode {
public:
  pAOSOAutostepGradNode (int _K);
  pAOSOAutostepGradNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pAOSOAutostepGradNode *parent_, *left_, *right_; //
  pAOSOAutostepGradSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds
  VecIdx sub_ci_; // subclass set
  pAOSOAutostepGradSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pAOSOAutostepGradNodeLess {
  bool operator () (const pAOSOAutostepGradNode* n1, const pAOSOAutostepGradNode* n2) {
    return n1->split_.expected_gain_ < 
      n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pAOSOAutostepGradNode*, 
                            std::vector<pAOSOAutostepGradNode*>, 
                            pAOSOAutostepGradNodeLess>
        QuepAOSOAutostepGradNode;

// Best Split Finder (helper class for parallel_reduce) 
class pAOSOAutostepGradTree;
struct pAOSOGrad_best_split_finder {
  pAOSOGrad_best_split_finder (pAOSOAutostepGradTree *_tree, 
    pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data); // customized data
  pAOSOGrad_best_split_finder (const pAOSOGrad_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pAOSOGrad_best_split_finder &rhs); // required

  pAOSOAutostepGradTree *tree_;
  pAOSOAutostepGradNode *node_;
  pAOSOAutostepGradData *data_;
  pAOSOAutostepGradSplit cb_split_;
};

// Tree
class pAOSOAutostepGradTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double ratio_si_, ratio_fi_;
    double weight_ratio_si_; 
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
  void split( pAOSOAutostepGradData* _data );
  void fit ( pAOSOAutostepGradData* _data );

  pAOSOAutostepGradNode* get_node (float* _sample);
  void get_is_leaf (VecInt& is_leaf);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample_samples (pAOSOAutostepGradData* _data);
  void subsample_classes_for_node (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data);

  void clear ();
  void creat_root_node (pAOSOAutostepGradData* _data);

  virtual bool find_best_candidate_split (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data);
  virtual bool find_best_split_num_var (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data, int _ivar,
    pAOSOAutostepGradSplit &spl);

  void make_node_sorted_idx(pAOSOAutostepGradNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pAOSOAutostepGradNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pAOSOAutostepGradSplit &cb_split);

  bool can_split_node (pAOSOAutostepGradNode* _node);
  bool split_node (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data);
  void calc_gain (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data);
  virtual void fit_node (pAOSOAutostepGradNode* _node, pAOSOAutostepGradData* _data);

protected:
  std::list<pAOSOAutostepGradNode> nodes_; // all nodes
  QuepAOSOAutostepGradNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes
};

// vector of pAOSOAutostepGradTree
typedef std::vector<pAOSOAutostepGradTree> VecpAOSOAutostepGradTree;

// Boost
class pAOSOGradAutostepBoost {
public:
  static const double EPS_LOSS;
  static const double MAX_F;

public:
  struct Param {
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
    double ratio_si_, ratio_fi_; // ratios for subsampling
    double weight_ratio_si_; // weight ratios for subsampling
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

public:
  std::vector<double> abs_grad_; // total gradient. indicator for stopping. 
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
  cv::Mat_<double> abs_grad_class_; // #iteration * #classes
  cv::Mat_<double> loss_class_;


protected:
  void train_init (MLData* _data);
  void update_F(int t);
  void update_p();
  void update_gg ();
  void update_abs_grad_class (int t);

  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  void calc_loss_class (MLData* _data, int t);

  bool should_stop (int t);
  void calc_grad( int t );


protected:
  int K_; // class count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  cv::Mat_<double> gg_; // gradient. #samples * #classes

  int NumIter_; // actual iteration number
  pAOSOAutostepGradData logitdata_;
  VecpAOSOAutostepGradTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
