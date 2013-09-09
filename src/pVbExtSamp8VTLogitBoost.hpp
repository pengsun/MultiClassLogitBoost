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
struct pVbExtSamp8VTData {
  MLData* data_cls_;
  cv::Mat_<double> *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
  cv::Mat_<double> *gg_; // #samples * 1
};

// Solver
struct pVbExtSamp8VTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  pVbExtSamp8VTSolver () {};
  pVbExtSamp8VTSolver (pVbExtSamp8VTData* _data, VecIdx* _ci);
  void set_data (pVbExtSamp8VTData* _data, VecIdx* _ci);

  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  std::vector<double> mg_, h_;
  pVbExtSamp8VTData* data_;
  VecIdx *ci_;
};

// Split descriptor
struct pVbExtSamp8VTSplit {
public:
  pVbExtSamp8VTSplit ();
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
struct pVbExtSamp8VTNode {
public:
  pVbExtSamp8VTNode (int _K);
  pVbExtSamp8VTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
  std::vector<double> fitvals_;

  int id_; // node ID. 0 for root node
  pVbExtSamp8VTNode *parent_, *left_, *right_; //
  pVbExtSamp8VTSplit split_;

  VecIdx sample_idx_; // for all the examples this node holds
  VecIdx sub_ci_; // subclass set
  pVbExtSamp8VTSolver sol_this_; // for all the examples this node holds
};

// Node Comparator: the less the expected gain, the less the node
struct pVbExtSamp8VTNodeLess {
  bool operator () (const pVbExtSamp8VTNode* n1, const pVbExtSamp8VTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for node
typedef std::priority_queue<pVbExtSamp8VTNode*, 
                            std::vector<pVbExtSamp8VTNode*>, 
                            pVbExtSamp8VTNodeLess>
        QuepVbExtSamp8VTNode;

// Best Split Finder (helper class for parallel_reduce) 
class pVbExtSamp8VTTree;
struct pVbExt8VT_best_split_finder {
  pVbExt8VT_best_split_finder (pVbExtSamp8VTTree *_tree, 
    pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data); // customized data
  pVbExt8VT_best_split_finder (const pVbExt8VT_best_split_finder &f, cv::Split); // required

  void operator () (const cv::BlockedRange &r); // required
  void join (pVbExt8VT_best_split_finder &rhs); // required

  pVbExtSamp8VTTree *tree_;
  pVbExtSamp8VTNode *node_;
  pVbExtSamp8VTData *data_;
  pVbExtSamp8VTSplit cb_split_;
};

// Tree
class pVbExtSamp8VTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    double ratio_si_, ratio_fi_, ratio_ci_;
    Param ();
  };
  Param param_;

public:
  int nr_wts, nr_wtc, nr_wtsc; // remaining number after weight trimming
                               // samples, classes, samples & classes

public:
  // TODO: Make sure of Thread Safety!
  VecIdx sub_si_, sub_fi_, sub_ci_; // sample/feature/class index after subsample 

public:
  void split( pVbExtSamp8VTData* _data );
  void fit ( pVbExtSamp8VTData* _data );

  pVbExtSamp8VTNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

public:
  void subsample (pVbExtSamp8VTData* _data);
  void subsample_classes_for_node (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data);

  void clear ();
  void creat_root_node (pVbExtSamp8VTData* _data);

  virtual bool find_best_candidate_split (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data);
  virtual bool find_best_split_num_var (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data, int _ivar,
                                        pVbExtSamp8VTSplit &spl);

  void make_node_sorted_idx(pVbExtSamp8VTNode* _node, MLData* _data, int _ivar, VecIdx& sorted_idx_node);
  bool set_best_split_num_var ( pVbExtSamp8VTNode* _node, MLData* _data, int _ivar, 
    VecIdx& node_sample_si,
    int best_i, double best_gain, double best_gain_left, double best_gain_right,
    pVbExtSamp8VTSplit &cb_split);

  bool can_split_node (pVbExtSamp8VTNode* _node);
  bool split_node (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data);
  void calc_gain (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data);
  virtual void fit_node (pVbExtSamp8VTNode* _node, pVbExtSamp8VTData* _data);

protected:
  std::list<pVbExtSamp8VTNode> nodes_; // all nodes
  QuepVbExtSamp8VTNode candidate_nodes_; // priority queue of candidate leaves for splitting
  // cb: current best
  // caching internal data, used by find_best_split*
  int K_; // #classes


};

// vector of pVbExtSamp8VTTree
typedef std::vector<pVbExtSamp8VTTree> VecpVbExtSamp8VTTree;

// Boost
class pVbExtSamp8VTLogitBoost {
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
  void get_nr (VecIdx& nr_wts, VecIdx& nr_wtc);

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
  pVbExtSamp8VTData logitdata_;
  VecpVbExtSamp8VTTree trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class
};
