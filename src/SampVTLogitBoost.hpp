#ifndef VTLogitBoost_h__
#define VTLogitBoost_h__

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>

// Vector for index
typedef std::vector<int> VecIdx;

// data shared by tree and booster
struct SampVTData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
};

// Adaptive One-vs-One Solver
struct SampVTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  SampVTSolver () {};
  SampVTSolver (SampVTData* _data);

  void set_data (SampVTData* _data);
  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  SampVTData* data_;
};

// Split descriptor
struct SampVTSplit {
public:
  SampVTSplit ();
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
struct SampVTNode {
public:
  SampVTNode (int _K);
  SampVTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  SampVTNode *parent_, *left_, *right_; //
  SampVTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds

  SampVTSolver sol_this_; // for all the examples this node holds
};


// Node Comparator: the less the expected gain, the less the node
struct SampVTNodeLess {
  bool operator () (const SampVTNode* n1, const SampVTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for aoso node
typedef std::priority_queue<SampVTNode*, 
                            std::vector<SampVTNode*>, 
                            SampVTNodeLess>
        QueSampVTNode;

// Tree
class SampVTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    Param ();
  };
  Param param_;

public:
  void split( SampVTData* _data );
  void fit ( SampVTData* _data );

  SampVTNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

protected:
  void clear ();
  void creat_root_node (SampVTData* _data);

  virtual bool find_best_candidate_split (SampVTNode* _node, SampVTData* _data);
  virtual bool find_best_split_num_var (SampVTNode* _node, SampVTData* _data, int _ivar);

  void make_node_sorted_idx(SampVTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( SampVTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right);

  bool can_split_node (SampVTNode* _node);
  bool split_node (SampVTNode* _node, SampVTData* _data);
  void calc_gain (SampVTNode* _node, SampVTData* _data);
  virtual void fit_node (SampVTNode* _node, SampVTData* _data);

protected:
  std::list<SampVTNode> nodes_; // all nodes
  QueSampVTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  SampVTSplit cb_split_, cvb_split_;
  int K_; // #classes
};

// vector of SampVTTree
typedef std::vector<SampVTTree> VecSampVTTree;

// Boost
class SampVTLogitBoost {
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
  SampVTData klogitdata_;
  VecSampVTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
#endif // VTLogitBoost_h__