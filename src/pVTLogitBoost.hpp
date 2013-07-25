#ifndef VTLogitBoost_h__
#define VTLogitBoost_h__

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>
#include <opencv2/core/internal.hpp>


// Vector for index
typedef std::vector<int> VecIdx;


// data shared by tree and booster
struct pVTLogitData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
};


// Adaptive One-vs-One Solver
struct pVTLogitSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pVTLogitSolver () {};
  pVTLogitSolver (pVTLogitData* _data);

  void set_data (pVTLogitData* _data);
  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pVTLogitData* data_;
};

// Split descriptor
struct pVTLogitSplit {
public:
  pVTLogitSplit ();
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
struct pVTLogitNode {
public:
  pVTLogitNode (int _K);
  pVTLogitNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pVTLogitNode *parent_, *left_, *right_; //
  pVTLogitSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds

  pVTLogitSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pVTLogitNodeLess {
  bool operator () (const pVTLogitNode* n1, const pVTLogitNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for aoso node
typedef std::priority_queue<pVTLogitNode*, 
                            std::vector<pVTLogitNode*>, 
                            pVTLogitNodeLess>
        QuepVTLogitNode;

// Best Split Finder (helper class for parallel_reduce) 
class pVTLogitTree;
struct pVT_best_split_finder {
  pVT_best_split_finder (pVTLogitTree *_tree, 
    pVTLogitNode* _node, pVTLogitData* _data); // customized data
  pVT_best_split_finder (const pVT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pVT_best_split_finder &rhs); // required

  pVTLogitTree *tree_;
  pVTLogitNode *node_;
  pVTLogitData *data_;
  pVTLogitSplit cb_split_;
};

// AOSO Tree
class pVTLogitTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    Param ();
  };
  Param param_;

public:
  void split( pVTLogitData* _data );
  void fit ( pVTLogitData* _data );

  pVTLogitNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void clear ();
  void creat_root_node (pVTLogitData* _data);

  virtual bool find_best_candidate_split (pVTLogitNode* _node, pVTLogitData* _data);
  virtual bool find_best_split_num_var (pVTLogitNode* _node, pVTLogitData* _data, int _ivar,
                                        pVTLogitSplit &spl);

  void make_node_sorted_idx(pVTLogitNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pVTLogitNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pVTLogitSplit &cb_split);

  bool can_split_node (pVTLogitNode* _node);
  bool split_node (pVTLogitNode* _node, pVTLogitData* _data);
  void calc_gain (pVTLogitNode* _node, pVTLogitData* _data);
  virtual void fit_node (pVTLogitNode* _node, pVTLogitData* _data);

protected:
  std::list<pVTLogitNode> nodes_; // all nodes
  QuepVTLogitNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes
};

// vector of pVTLogitTree
typedef std::vector<pVTLogitTree> VecpVTLogitTree;

// AOTO Boost
class pVTLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAX_F;

public:
  struct Param {
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
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
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  pVTLogitData klogitdata_;
  VecpVTLogitTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
#endif // VTLogitBoost_h__