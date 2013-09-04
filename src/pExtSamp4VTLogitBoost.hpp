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
struct pExtSamp4VTData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
  cv::Mat_<double> *nd_; // #samples * 1
};

// Solver
struct pExtSamp4VTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pExtSamp4VTSolver () {};
  pExtSamp4VTSolver (pExtSamp4VTData* _data, VecIdx* _ci);
  void set_data (pExtSamp4VTData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pExtSamp4VTData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pExtSamp4VTSplit {
public:
  pExtSamp4VTSplit ();
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
struct pExtSamp4VTNode {
public:
  pExtSamp4VTNode (int _K);
  pExtSamp4VTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pExtSamp4VTNode *parent_, *left_, *right_; //
  pExtSamp4VTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds

  pExtSamp4VTSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pExtSamp4VTNodeLess {
  bool operator () (const pExtSamp4VTNode* n1, const pExtSamp4VTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pExtSamp4VTNode*, 
                            std::vector<pExtSamp4VTNode*>, 
                            pExtSamp4VTNodeLess>
        QuepExtSamp4VTNode;

// Best Split Finder (helper class for parallel_reduce) 
class pExtSamp4VTTree;
struct pExt4VT_best_split_finder {
  pExt4VT_best_split_finder (pExtSamp4VTTree *_tree, 
    pExtSamp4VTNode* _node, pExtSamp4VTData* _data); // customized data
  pExt4VT_best_split_finder (const pExt4VT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pExt4VT_best_split_finder &rhs); // required

  pExtSamp4VTTree *tree_;
  pExtSamp4VTNode *node_;
  pExtSamp4VTData *data_;
  pExtSamp4VTSplit cb_split_;
};

// Tree
class pExtSamp4VTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double ratio_si_, ratio_fi_, ratio_ci_;
    Param ();
  };
  Param param_;

public:
  void split( pExtSamp4VTData* _data );
  void fit ( pExtSamp4VTData* _data );

  pExtSamp4VTNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample (pExtSamp4VTData* _data);

  void clear ();
  void creat_root_node (pExtSamp4VTData* _data);

  virtual bool find_best_candidate_split (pExtSamp4VTNode* _node, pExtSamp4VTData* _data);
  virtual bool find_best_split_num_var (pExtSamp4VTNode* _node, pExtSamp4VTData* _data, int _ivar,
                                        pExtSamp4VTSplit &spl);

  void make_node_sorted_idx(pExtSamp4VTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pExtSamp4VTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pExtSamp4VTSplit &cb_split);

  bool can_split_node (pExtSamp4VTNode* _node);
  bool split_node (pExtSamp4VTNode* _node, pExtSamp4VTData* _data);
  void calc_gain (pExtSamp4VTNode* _node, pExtSamp4VTData* _data);
  virtual void fit_node (pExtSamp4VTNode* _node, pExtSamp4VTData* _data);

protected:
  std::list<pExtSamp4VTNode> nodes_; // all nodes
  QuepExtSamp4VTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes

public:
  VecIdx sub_si_, sub_fi_, sub_ci_; // sample/feature/class index after subsample 
};

// vector of pExtSamp4VTTree
typedef std::vector<pExtSamp4VTTree> VecpExtSamp4VTTree;

// Boost
class pExtSamp4VTLogitBoost {
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
  void update_nd();

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
  cv::Mat_<double> nd_; // Newton decrement. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  pExtSamp4VTData logitdata_;
  VecpExtSamp4VTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
