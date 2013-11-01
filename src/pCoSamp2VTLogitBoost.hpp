#pragma once

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>
#include <opencv2/core/internal.hpp>

// typedefs
typedef std::vector<int>    VecIdx;
typedef std::vector<VecIdx > VecVecIdx;
typedef std::vector<int>    VecInt;
typedef std::vector<double> VecDbl;
typedef cv::Mat_<double>    MatDbl;

// data shared by tree and booster
struct pCoSampVT2Data {
  MLData* data_cls_;
  MatDbl *p_; // #samples * #class
  MatDbl *L_; // #samples * 1
  MatDbl *ww_; // #samples * #class
};

// Solver
struct pCoSamp2VT2Solver {
  static const double MAXGAMMA;

  pCoSamp2VT2Solver () {};
  pCoSamp2VT2Solver (pCoSampVT2Data* _data, VecVecIdx* _sitoci);
  void set_data   (pCoSampVT2Data* _data, VecVecIdx* _sitoci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pCoSampVT2Data* data_;
  VecVecIdx *sitoci_;

  double cur_gain_;
  VecDbl cur_gain_cls_;
};

// Split descriptor
struct pCoSampVT2Split {
public:
  pCoSampVT2Split ();
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
struct pCoSampVT2Node {
public:
  pCoSampVT2Node (int _K);
  pCoSampVT2Node (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pCoSampVT2Node *parent_, *left_, *right_; //
  pCoSampVT2Split split_;

  VecIdx sample_idx_; // for all the examples this node holds
  pCoSamp2VT2Solver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pCoSampVT2NodeLess {
  bool operator () (const pCoSampVT2Node* n1, const pCoSampVT2Node* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pCoSampVT2Node*, 
                            std::vector<pCoSampVT2Node*>, 
                            pCoSampVT2NodeLess>
        QuepCoSampVT2Node;

// Best Split Finder (helper class for parallel_reduce) 
class pCoSampVT2Tree;
struct pCoSampVT_best_split_finder {
  pCoSampVT_best_split_finder (pCoSampVT2Tree *_tree, 
    pCoSampVT2Node* _node, pCoSampVT2Data* _data); // customized data
  pCoSampVT_best_split_finder (const pCoSampVT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pCoSampVT_best_split_finder &rhs); // required

  pCoSampVT2Tree *tree_;
  pCoSampVT2Node *node_;
  pCoSampVT2Data *data_;
  pCoSampVT2Split cb_split_;
};

// Tree
class pCoSampVT2Tree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double rb_, rf_;
    double wrb_; 
    Param ();
  };
  Param param_;

public:
  // TODO: Make sure of Thread Safety!
  VecIdx sub_si_, sub_fi_; // sample/feature/class index after subsample_samples 
  VecVecIdx si_to_ci_;

public:
  VecDbl node_cc_; // class count for each node   
  VecInt node_sc_; // sample count for each node

public:
  void split( pCoSampVT2Data* _data );
  void fit ( pCoSampVT2Data* _data );

  pCoSampVT2Node* get_node (float* _sample);
  void get_is_leaf (VecInt& is_leaf);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
#if 0
  void subsample_samples (pCoSampVTData* _data);
  void subsample_classes_for_node (pCoSampVTNode* _node, pCoSampVTData* _data);
#endif
  void subsample_budget (pCoSampVT2Data* _data);

  void clear ();
  void creat_root_node (pCoSampVT2Data* _data);

  virtual bool find_best_candidate_split (pCoSampVT2Node* _node, pCoSampVT2Data* _data);
  virtual bool find_best_split_num_var (pCoSampVT2Node* _node, pCoSampVT2Data* _data, int _ivar,
                                        pCoSampVT2Split &spl);

  void make_node_sorted_idx(pCoSampVT2Node* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pCoSampVT2Node* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pCoSampVT2Split &cb_split);

  bool can_split_node (pCoSampVT2Node* _node);
  bool split_node (pCoSampVT2Node* _node, pCoSampVT2Data* _data);
  void calc_gain (pCoSampVT2Node* _node, pCoSampVT2Data* _data);
  virtual void fit_node (pCoSampVT2Node* _node, pCoSampVT2Data* _data);

protected:
  std::list<pCoSampVT2Node> nodes_; // all nodes
  QuepCoSampVT2Node candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes
};

// vector of pCoSampVT2Tree
typedef std::vector<pCoSampVT2Tree> VecpCoSampVT2Tree;

// Boost
class pCoSamp2VTLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAX_F;

public:
  struct Param {
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
    double rb_, rf_; // ratios of budget, features
    double wrb_; // weight ratios of budget
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
  void get_cc (int itree, VecDbl& node_cc);
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
  void update_ww ();
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
  cv::Mat_<double> ww_; // gradient. #samples * #classes

  int NumIter_; // actual iteration number
  pCoSampVT2Data logitdata_;
  VecpCoSampVT2Tree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
