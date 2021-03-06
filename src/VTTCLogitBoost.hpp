#ifndef VTTCLogitBoost_h__
#define VTTCLogitBoost_h__

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>

// data shared by tree and booster
struct VTTCLogitData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
};

// Vector for index
typedef std::vector<int> VecIdx;

// Split descriptor
struct VTTCLogitSplit {
public:
  VTTCLogitSplit ();
  void reset ();

  int var_idx_; // variable for split
  VAR_TYPE var_type_; // variable type

  float threshold_; // for numeric variable 
  std::bitset<MLData::MAX_VAR_CAT> subset_; // mask for category variable, maximum 64 #category 

  double this_gain_;
  double expected_gain_;
  double left_node_gain_, right_node_gain_;
};

// Node. Vector value
struct VTTCLogitNode {
public:
  VTTCLogitNode (int _K);
  VTTCLogitNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_, fitvals_prev_; // current, previous node values

  int id_; // node ID. 0 for root node
  VTTCLogitNode *parent_, *left_, *right_; //
  VTTCLogitSplit split_;

  VecIdx sample_idx_; // for training
};

// Node Comparator: the less the expected gain, the less the node
struct VTTCLogitNodeLess {
  bool operator () (const VTTCLogitNode* n1, const VTTCLogitNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for aoso node
typedef std::priority_queue<VTTCLogitNode*, 
                            std::vector<VTTCLogitNode*>, 
                            VTTCLogitNodeLess>
        QueVTTCLogitNode;

// Solver (K-vector)
struct VTTCLogitSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  VTTCLogitSolver (VTTCLogitData* _data);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);
  
  void calc_gamma (const double* init_gamma, double lambda, double* gamma);
  void calc_gain (double& gain);

public:
	std::vector<double> mg_, h_;
  VTTCLogitData* data_;
};

// Regression Tree
class VTTCLogitTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double lambda_; // tuning parameter for L1 regularization
    Param ();
  };
  Param param_;

public:
  void split( VTTCLogitData* _data );
  void fit ( VTTCLogitData* _data );

  VTTCLogitNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

  void predict_prev_val (MLData* _data);
  void predict_prev_val (float* _sample, float* _score);

protected:
  void clear ();
  void creat_root_node (VTTCLogitData* _data);

  virtual bool find_best_candidate_split (VTTCLogitNode* _node, VTTCLogitData* _data);
  virtual bool find_best_split_num_var (VTTCLogitNode* _node, VTTCLogitData* _data, int _ivar);

  void make_node_sorted_idx(VTTCLogitNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( VTTCLogitNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right);

  bool can_split_node (VTTCLogitNode* _node);
  bool split_node (VTTCLogitNode* _node, VTTCLogitData* _data);
  void calc_gain (VTTCLogitNode* _node, VTTCLogitData* _data);
  virtual void fit_node (VTTCLogitNode* _node, VTTCLogitData* _data);



protected:
  std::list<VTTCLogitNode> nodes_; // all nodes
  QueVTTCLogitNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  VTTCLogitSplit cb_split_, cvb_split_;
  int K_; // #classes
};

// vector of VTTCLogitTree
typedef std::vector<VTTCLogitTree> VecVTTCLogitTree;

// Boost
class VTTCLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAX_F;

public:
  struct Param {
    int T;     // max iterations
    double lambda;  // magnitude for L1 regularizer
    int J;     // #terminal nodes
    int ns;    // node size
    int TRound;     // number of rounds for cyclical coordinate descent
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
  void update_F_using_tree(int t);
  void update_p();

  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  void calc_grad (int t);

  bool should_stop (int t);

  void update_all_tree_leaves (int t);

public:
  std::vector<double> abs_grad_; // total gradient. indicator for stopping.
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class

protected:
  int K_; // class count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  VTTCLogitData logitdata_;
  VecVTTCLogitTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
#endif // VTTCLogitBoost_h__