#ifndef AOTOBoostSol2Sel2gain_h__
#define AOTOBoostSol2Sel2gain_h__

#include "MLData.hpp"
#include <vector>
#include <list>
#include <queue>
#include <bitset>

// data shared by tree and booster
struct ConsVTData {
  MLData* data_cls_;
  cv::Mat_<double> *F_, *p_; // #samples * #class
  cv::Mat_<double> *L_; // #samples * 1
};

// Vector for index
typedef std::vector<int> VecIdx;

// Split descriptor
struct ConsVTSplit {
public:
  ConsVTSplit ();
  void reset ();

  int var_idx_; // variable for split
  VAR_TYPE var_type_; // variable type
  float threshold_; // for numeric variable 

  double this_gain_;
  double expected_gain_;
};

// AOTO Node. Vector value
struct ConsVTNode {
public:
  ConsVTNode (int _K);
  ConsVTNode (int _id, int _K);
  // to which side should the sample be sent. -1:left, +1:right
  int calc_dir (float* _psample);

public:
	bool split_succeed_;

  std::vector<double> fitvals_;

	int id_; // node ID. 0 for root node
  ConsVTNode *parent_, *left_, *right_; //
  ConsVTSplit split_;

	VecIdx sample_idx_; // for training
};

// Node Comparator: the less the expected gain, the less the node
struct ConsVTNodegainLess {
  bool operator () (const ConsVTNode* n1, const ConsVTNode* n2) {
    return n1->split_.expected_gain_ < 
           n2->split_.expected_gain_;
  }
};

// Priority queue for aoto node
typedef std::priority_queue<ConsVTNode*, 
                            std::vector<ConsVTNode*>, 
                            ConsVTNodegainLess>
                            QueConsVTNode;
// Adaptive One-to-One Solver
struct ConsVTSolver {
  static const double MAXGAMMA;
  //static const double EPS;

  ConsVTSolver () {};
  ConsVTSolver (ConsVTData* _data);

  void set_data (ConsVTData* _data);
  void update_internal (VecIdx& vidx);
  void update_internal_incre (int idx);
  void update_internal_decre (int idx);

  void calc_gamma (double* gamma);
  void calc_gain (double& gain);

public:
  int n_;
  std::vector<double> mg_, h_;
  ConsVTData* data_;
};

// AOTO Tree
class ConsVTTree {
public:
  struct Param {
    int max_leaves_; // maximum leaves (terminal nodes)
    int node_size_;   // minimum sample size in leaf
    Param ():max_leaves_(2), node_size_(5){}
  };
  Param param_;

  std::list<ConsVTNode> nodes_; // all nodes
	int K_;
	int N_;

public:
  ConsVTTree():K_(0), N_(0){}
  void split( ConsVTData* _data );
  void fit ( ConsVTData* _data );

  ConsVTNode* get_node (float* _sample);
  void predict (MLData* _data);
  void predict (float* _sample, float* _score);

protected:
  void clear ();
  void creat_root_node (ConsVTData* _data);

  virtual bool find_best_candidate_split (std::vector<ConsVTNode*> _node_buffer, ConsVTData* _data);
  void build_lookup_table(std::vector<ConsVTNode*> _node_buffer, std::vector<int>& _lookup_table);
  bool can_split_node (ConsVTNode* _node);
  bool split_node (ConsVTNode* _node, ConsVTData* _data);
  void init_node (ConsVTNode* _node, ConsVTData* _data);
};

// the struct to save Tree into files:static storage tree
struct StorTree{
	cv::Mat_<int> nodes_;
	cv::Mat_<double> splits_;
	cv::Mat_<double> leaves_;
};

// AOTO Boost
class ConsVTLogitBoost {
public:
  static const double EPS_LOSS;
  static const double MAXF;

public:
  struct Param{
    int T;     // max iterations
    double v;  // shrinkage
    int J;     // #terminal nodes
    int ns;    // node size
  };
  Param param_;
  
public:
  void train (MLData* _data);

  void predict (MLData* _data);
  virtual void predict (float* _sapmle, float* _score);
  void predict (MLData* _data, int _Tpre);
  virtual void predict (float* _sapmle, float* _score, int _Tpre);

  void convertToStorTrees();

  int get_class_count ();
  int get_num_iter ();
  double get_train_loss ();

protected:
  void train_init (MLData* _data);
  void update_F(int t);
  void calc_p ();
  void calc_loss (MLData* _data);
  void calc_loss_iter (int t);
  bool should_stop (int t);
  void calc_grad(int t);
  void convert(ConsVTNode * _root_Node, 
				StorTree & staTree, int& _leafId, int& _splitId);

protected:
  int K_; // class count
  int N_; // sample count
  cv::Mat_<double> L_; // Loss. #samples
  cv::Mat_<double> L_iter_; // Loss. #iteration
  int NumIter_; // actual iteration number
  ConsVTData aotodata_;
  std::vector<ConsVTTree> trees_;

  int Tpre_beg_; // Beginning tree for test data
  cv::Mat_<double> Fpre_; // Score for test data. #samples * #class

public:
  cv::Mat_<double> F_, p_; // Score and Probability. #samples * #class
	std::vector<double> abs_grad_; // total gradient.
	std::vector<StorTree> stor_Trees_;
};



#endif // AOTOBoostSol2Sel2_h__