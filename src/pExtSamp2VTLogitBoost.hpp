#pragma once

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>
#include <opencv2/core/internal.hpp>

// Vector for index
typedef std::vector<int> VecIdx;
typedef std::vector<double> VecDbl;

// data shared by tree and booster
struct pExtSamp2VTData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
  cv::Mat_<double> *g_; // #samples * 1
};

// Solver
struct pExtSamp2VTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pExtSamp2VTSolver () {};
  pExtSamp2VTSolver (pExtSamp2VTData* _data, VecIdx* _ci);
  void set_data (pExtSamp2VTData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pExtSamp2VTData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pExtSamp2VTSplit {
public:
  pExtSamp2VTSplit ();
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
struct pExtSamp2VTNode {
public:
  pExtSamp2VTNode (int _K);
  pExtSamp2VTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pExtSamp2VTNode *parent_, *left_, *right_; //
  pExtSamp2VTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds

  pExtSamp2VTSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pExtSamp2VTNodeLess {
  bool operator () (const pExtSamp2VTNode* n1, const pExtSamp2VTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pExtSamp2VTNode*, 
                            std::vector<pExtSamp2VTNode*>, 
                            pExtSamp2VTNodeLess>
        QuepExtSamp2VTNode;

// Best Split Finder (helper class for parallel_reduce) 
class pExtSamp2VTTree;
struct pExt2VT_best_split_finder {
  pExt2VT_best_split_finder (pExtSamp2VTTree *_tree, 
    pExtSamp2VTNode* _node, pExtSamp2VTData* _data); // customized data
  pExt2VT_best_split_finder (const pExt2VT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pExt2VT_best_split_finder &rhs); // required

  pExtSamp2VTTree *tree_;
  pExtSamp2VTNode *node_;
  pExtSamp2VTData *data_;
  pExtSamp2VTSplit cb_split_;
};

// Tree
class pExtSamp2VTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double ratio_si_, ratio_fi_, ratio_ci_;
    Param ();
  };
  Param param_;

public:
  void split( pExtSamp2VTData* _data );
  void fit ( pExtSamp2VTData* _data );

  pExtSamp2VTNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample (pExtSamp2VTData* _data);

  void clear ();
  void creat_root_node (pExtSamp2VTData* _data);

  virtual bool find_best_candidate_split (pExtSamp2VTNode* _node, pExtSamp2VTData* _data);
  virtual bool find_best_split_num_var (pExtSamp2VTNode* _node, pExtSamp2VTData* _data, int _ivar,
                                        pExtSamp2VTSplit &spl);

  void make_node_sorted_idx(pExtSamp2VTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pExtSamp2VTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pExtSamp2VTSplit &cb_split);

  bool can_split_node (pExtSamp2VTNode* _node);
  bool split_node (pExtSamp2VTNode* _node, pExtSamp2VTData* _data);
  void calc_gain (pExtSamp2VTNode* _node, pExtSamp2VTData* _data);
  virtual void fit_node (pExtSamp2VTNode* _node, pExtSamp2VTData* _data);

protected:
  std::list<pExtSamp2VTNode> nodes_; // all nodes
  QuepExtSamp2VTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes

public:
  VecIdx sub_si_, sub_fi_, sub_ci_; // sample/feature/class index after subsample 
};

// vector of pExtSamp2VTTree
typedef std::vector<pExtSamp2VTTree> VecpExtSamp2VTTree;

// Boost
class pExtSamp2VTLogitBoost {
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

protected:
  void train_init (MLData* _data);
  void update_F(int t);
  void update_p();
  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  bool should_stop (int t);
  void calc_grad (int t);

public:
  std::vector<double> abs_grad_; // total gradient. indicator for stopping. 
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
protected:
  int K_; // class count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> g_; // gradient. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  pExtSamp2VTData logitdata_;
  VecpExtSamp2VTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
