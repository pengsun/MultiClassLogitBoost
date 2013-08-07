#include "ConsVTLogitBoost.hpp"
#include <iostream>

using namespace std;
using namespace cv;

// Implementation of ConsVTSplit
ConsVTSplit::ConsVTSplit()
{
  reset();
}

void ConsVTSplit::reset()
{
  var_idx_ = -1;
  threshold_ = FLT_MAX;
  this_gain_ = -1;
  expected_gain_ = 0;
}

// Implementation of ConsVTNode
ConsVTNode::ConsVTNode(int _K)
{
  id_ = 0;
  split_succeed_ = false;
  parent_ = left_ = right_ = 0;

  fitvals_.assign(_K, 0);
}

ConsVTNode::ConsVTNode( int _id, int _K )
{
  id_ = _id;
  split_succeed_ = false;
  parent_ = left_ = right_ = 0;

  fitvals_.assign(_K, 0);
}

int ConsVTNode::calc_dir( float* _psample )
{
  float _val = *(_psample + split_.var_idx_);

  int dir = 0;
  if (split_.var_type_==VAR_CAT) {
    // TODO: raise an error
    /*
    int tmp = int(_val);
    dir = ( split_.subset_[tmp] == true ) ? (-1) : (+1);
    */
  }
  else { // split_.var_type_==VAR_NUM
    dir = (_val < split_.threshold_)? (-1) : (+1); 
  }

  return dir;
}
// Implementation of ConsVTSolver
const double ConsVTSolver::MAXGAMMA = 5.0;
ConsVTSolver::ConsVTSolver( ConsVTData*  _data)
{ 
  set_data(_data);
}

void ConsVTSolver::set_data( ConsVTData*  _data)
{ 
  data_ = _data;

  int K = data_->data_cls_->get_class_count();
  mg_.assign(K, 0.0);
  h_.assign(K, 0.0);

  n_ = 0;
}

void ConsVTSolver::update_internal( VecIdx& vidx )
{
  for (VecIdx::iterator it = vidx.begin(); it!=vidx.end(); ++it  ) {
    int idx = *it;
    update_internal_incre(idx);
  } // for it
}

void ConsVTSolver::update_internal_incre( int idx )
{
  ++n_;

  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[k] += (1-pik);
    else mg_[k] += (-pik);

    h_[k] += pik*(1-pik);
  }
}

void ConsVTSolver::update_internal_decre( int idx )
{
  --n_;

  int yi = int( data_->data_cls_->Y.at<float>(idx) );
  double* ptr_pi = data_->p_->ptr<double>(idx);

  // mg and h
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double pik = *(ptr_pi + k);

    if (yi==k) mg_[k] -= (1-pik);
    else mg_[k] -= (-pik);

    h_[k] -= pik*(1-pik);
  }

}

void ConsVTSolver::calc_gamma( double *gamma)
{
  int K = mg_.size();
  for (int kk = 0; kk < K; ++kk) {
    double smg = mg_[kk];
    double sh  = h_[kk];
    //if (sh <= 0) cv::error("ConsVTSolver::calc_gamma: Invalid Hessian.");
    if (sh == 0) sh = 1;

    double sgamma = smg/sh;
    double cap = sgamma;
    if (cap<-MAXGAMMA) cap = -MAXGAMMA;
    else if (cap>MAXGAMMA) cap = MAXGAMMA;
    *(gamma+kk) = cap;

  }
}

void ConsVTSolver::calc_gain( double& gain )
{
  gain = 0;
  int KK = mg_.size();
  for (int k = 0; k < KK; ++k) {
    double smg = mg_[k];
    double sh  = h_[k];
    if (sh == 0) sh = 1;

    gain += (smg*smg/sh);
  }
  gain = 0.5*gain;
}

// Implementation of ConsVTTree
void ConsVTTree::split( ConsVTData* _data )
{
	K_ = _data->data_cls_->get_class_count();
	N_ = _data->data_cls_->X.rows;

	// priority queue of candidate leaves for splitting
	QueConsVTNode candidate_nodes; 
	std::vector<bool> split_sucess;
	std::vector<ConsVTNode*> node_buffer;

	int nleaves = 1;
	
	creat_root_node(_data);
	node_buffer.push_back(&nodes_.front());
	do 
	{
		bool flag = find_best_candidate_split(node_buffer,  _data);
		if(flag == false) break;

		for(int i=0; i<node_buffer.size(); i++){
			  candidate_nodes.push(node_buffer[i]);
		}
		node_buffer.clear();

		while (!candidate_nodes.empty())
		{
			// the most prior node
			ConsVTNode* cur_node = candidate_nodes.top(); 
			candidate_nodes.pop();

			if (can_split_node(cur_node))
			{
				if(!cur_node->split_succeed_){
					// split nodes failed
					node_buffer.push_back(cur_node);
					continue;
				}

				// create left and right node
				int ID = nodes_.size();
				ConsVTNode tmp1(ID, K_);
				ConsVTNode tmp2(++ID, K_);

				// send each sample to left/right node
				int nn = cur_node->sample_idx_.size();
				MLData* data_cls = _data->data_cls_;
				for (int i = 0; i < nn; ++i) {
					int idx = cur_node->sample_idx_[i];

					float* pSample = (float*)data_cls->X.ptr(idx);
					int dir = cur_node->calc_dir(pSample);
					if (dir == -1) 
						tmp1.sample_idx_.push_back(idx);
					else 
						tmp2.sample_idx_.push_back(idx);
				}


				ConsVTNode *pTmp;
				//left node
				nodes_.push_back(tmp1);
				pTmp = &(nodes_.back());
				init_node(pTmp, _data);

				cur_node->left_ = pTmp;
				pTmp->parent_ = cur_node;

				node_buffer.push_back(pTmp);

				//right node
				nodes_.push_back(tmp2);
				pTmp = &(nodes_.back());
				init_node(pTmp, _data);

				cur_node->right_ = pTmp;
				pTmp->parent_ = cur_node;

				node_buffer.push_back(pTmp);

				// no longer used in later splitting
				// release memory.
				VecIdx tmp;
				tmp.swap(cur_node->sample_idx_); 

				nleaves ++;
				if(nleaves >= param_.max_leaves_) break;

			}
		}

	} while (nleaves < param_.max_leaves_ && !node_buffer.empty());

}

void ConsVTTree::creat_root_node( ConsVTData* _data )
{
  nodes_.push_back( ConsVTNode(0,K_) );
  ConsVTNode* root = &(nodes_.back());

  // samples in node
  root->sample_idx_.resize(N_);
  for (int i = 0; i < N_; ++i) {
    root->sample_idx_[i] = i;
  }

  // cls1, cls2, this_gain
  this->init_node(root, _data);
}
bool ConsVTTree::find_best_candidate_split(std::vector<ConsVTNode*> _node_buffer, ConsVTData* _data)
{
	MLData* data_cls = _data->data_cls_;
	int nvar = data_cls->X.cols;
	int buffer_size = _node_buffer.size();

	//build lookup table
	std::vector<int> lookup_table;
	build_lookup_table(_node_buffer, lookup_table);

	//initialize node_gain
	std::vector<double> left_node_gain, right_node_gain;
	left_node_gain.resize(buffer_size, 0);
	right_node_gain.resize(buffer_size, 0);

	double GAIN = 0;
	std::vector<ConsVTSolver> SOL_L, SOL_R;
	for (int k=0; k < buffer_size; k++){
		//set flag
		_node_buffer[k]->split_succeed_ = false;		
		//initial GAIN of all nodes
		GAIN += _node_buffer[k]->split_.this_gain_;
		//initial ConsVTSolver
		SOL_L.push_back(ConsVTSolver(_data));
		SOL_R.push_back(ConsVTSolver(_data));
		SOL_R[k].update_internal(_node_buffer[k]->sample_idx_);
	}
	
	int best_var = -1;
	double best_gain = 0, best_threshold = -1;
	for (int ivar = 0; ivar < nvar; ivar++) 
	{
		if (_data->data_cls_->var_type[ivar] == VAR_CAT) {
			// raise an error
			continue;
		}
		else { // VAR_NUM
			int i;
			double curr_gain = GAIN;
 			std::vector<ConsVTSolver> sol_left = SOL_L;
			std::vector<ConsVTSolver> sol_right = SOL_R;		

			// initial gain
			for (i=0; i<buffer_size; i++)
			{
				left_node_gain[i] = 0;
				right_node_gain[i] = _node_buffer[i]->split_.this_gain_;
			}

			VecIdx16 sam_idx16;
			VecIdx32 sam_idx32;
			if (data_cls->is_idx16)	sam_idx16 = data_cls->var_num_sidx16[ivar];
			else sam_idx32 = data_cls->var_num_sidx32[ivar];

			for (i=0; i < N_-1; i++)
			{
				//for each index
				int idx, idx1;				
				if (data_cls->is_idx16){
					idx = sam_idx16[i];
					idx1 = sam_idx16[i+1];
				}					
				else{
					idx = sam_idx32[i];
					idx1 = sam_idx32[i+1];
				}

				//find the node which contains idx
				int j = lookup_table[idx];

				//update left & right
				if(j == -1) continue; // did not find
				else{
					sol_left[j].update_internal_incre(idx);
					sol_right[j].update_internal_decre(idx);

					// check left & right
					double gL0 , gL1;
					gL0 = left_node_gain[j];
					sol_left[j].calc_gain(gL1);
					left_node_gain[j] = gL1;

					double gR0, gR1;
					gR0 = right_node_gain[j];
					sol_right[j].calc_gain(gR1);
					right_node_gain[j] = gR1;

					curr_gain +=  gR1 + gL1 -gR0 - gL0;

					// skip if overlap
					float x1, x2;
					x1= data_cls->X.at<float>(idx, ivar);
					x2 = data_cls->X.at<float>(idx1, ivar);

					//set best value					
					if (x1 != x2 && curr_gain > best_gain) {
						best_gain = curr_gain;
						best_var = ivar;
						best_threshold = (x1 + x2)/2;
						for (int k=0; k<buffer_size; k++)
						{
							_node_buffer[k]->split_.expected_gain_ = right_node_gain[k] + left_node_gain[k] -
																	_node_buffer[j]->split_.this_gain_; 

							_node_buffer[k]->split_succeed_ = (sol_left[k].n_ >= param_.node_size_) &&
															 (sol_right[k].n_ >= param_.node_size_);
						}
					}		
				}//if
			}//for
		}
	}
	
	//set value
	for (int k=0; k<buffer_size; k++){
		_node_buffer[k]->split_.var_idx_ = best_var;
		_node_buffer[k]->split_.var_type_ = VAR_NUM;
		_node_buffer[k]->split_.threshold_ = best_threshold;		
	}

	//sucess?
	bool found_flag = false;
	for (int k=0; k<buffer_size; k++){
		if(_node_buffer[k]->split_succeed_){
			found_flag = true;
			break;
		}			
	}
	return found_flag;
}

void ConsVTTree::build_lookup_table(std::vector<ConsVTNode*> _node_buffer, std::vector<int>& _lookup_table)
{
	//bulid look up table
	int buffer_size = _node_buffer.size();
	_lookup_table.resize(N_, -1);

	int i, j, s;
	for (j = 0; j < buffer_size ; j++)
	{
		VecIdx sample_idx =  _node_buffer[j]->sample_idx_;
		s = sample_idx.size();
		for (i =0; i< s; i++)
		{
			int idx = sample_idx[i];
			_lookup_table[idx] = j;
		}
	}
}

bool ConsVTTree::can_split_node( ConsVTNode* _node )
{
  int nn = _node->sample_idx_.size();
  int idx = _node->split_.var_idx_;
  return (nn > param_.node_size_    && // large enough node size
          idx != -1);                  // has candidate split  
}

void ConsVTTree::init_node(ConsVTNode* _node, ConsVTData* _data)
{
	//set h, mg
	ConsVTSolver sol(_data);
	sol.update_internal(_node->sample_idx_);

	//set gain
	double gain;
	sol.calc_gain(gain);
	_node->split_.this_gain_ = gain;
}

void ConsVTTree::fit( ConsVTData* _data )
{
  // fitting node data for each leaf
  std::list<ConsVTNode>::iterator it;
  for (it = nodes_.begin(); it != nodes_.end(); ++it) {
    ConsVTNode* nd = &(*it);

    if (nd->left_!=0) { // not a leaf
      continue;
    } 

	int nn = nd->sample_idx_.size();
	CV_Assert(nn>0);

	ConsVTSolver sol(_data);
	sol.update_internal(nd->sample_idx_);
	sol.calc_gamma( &(nd->fitvals_[0]) );

    // release memory.
    // no longer used in later splitting
	VecIdx tmp;
	tmp.swap(nd->sample_idx_);
  }
}

ConsVTNode* ConsVTTree::get_node( float* _sample)
{
  ConsVTNode* cur_node = &(nodes_.front());
  while (true) {
    if (cur_node->left_==0) break; // leaf reached 

    int dir = cur_node->calc_dir(_sample);
    ConsVTNode* next = (dir==-1) ? (cur_node->left_) : (cur_node->right_);
    cur_node = next;
  }
  return cur_node;
}

void ConsVTTree::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void ConsVTTree::predict( float* _sample, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  }

  // update the two class
  ConsVTNode* nd;
  nd = get_node(_sample);

  for (int k = 0; k < K_; ++k) {
    *(_score + k) = static_cast<float>( nd->fitvals_[k] );
  }
}


//==============================================
// Implementation of ConsVTLogitBoost
const double ConsVTLogitBoost::EPS_LOSS = 1e-14;
const double ConsVTLogitBoost::MAXF = 100;
void ConsVTLogitBoost::train( MLData* _data )
{
	// K_, N_, F, p, Loss, Tree vector init
	train_init(_data);

  for (int t = 0; t < param_.T; ++t) {
    trees_[t].split(&aotodata_);
    trees_[t].fit(&aotodata_);

    update_F(t);
    calc_p();
    calc_loss(_data);
    calc_loss_iter(t);
	  calc_grad(t);

    NumIter_ = t + 1;
    if ( should_stop(t) ) break;
  } // for t

}

void ConsVTLogitBoost::predict( MLData* _data )
{
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);

  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score);
  }
}

void ConsVTLogitBoost::predict( float* _sapmle, float* _score )
{
  // initialize
  for (int k = 0; k < K_; ++k) {
    *(_score+k) = 0;
  } // for k

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = 0; t < NumIter_; ++t) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    } // for k
  } // for t
}

void ConsVTLogitBoost::predict( MLData* _data, int _Tpre )
{
  // trees to be used
  if (_Tpre > NumIter_) _Tpre = NumIter_;
  if (_Tpre < 1) _Tpre = 1; // _Tpre in [1,T]
  if (Tpre_beg_ > _Tpre) Tpre_beg_ = 0;

  // initialize predicted score
  int N = _data->X.rows;
  int K = K_;
  if (_data->Y.rows!=N || _data->Y.cols!=K)
    _data->Y.create(N,K);
  
  // initialize internal score if necessary
  if (Tpre_beg_ == 0) {
    Fpre_.create(N,K);
    Fpre_ = 0;
  }

  // for each sample
  for (int i = 0; i < N; ++i) {
    float* p = _data->X.ptr<float>(i);
    float* score = _data->Y.ptr<float>(i);
    predict(p,score, _Tpre);
    
    // update score and internal score Fpre_
    double* pp = Fpre_.ptr<double>(i);
    for (int k = 0; k < K; ++k) {
      *(score+k) += *(pp+k);
      *(pp+k) = *(score+k);
    }
  }
  
  // Set the new beginning tree
  Tpre_beg_ = _Tpre;

}

void ConsVTLogitBoost::predict( float* _sapmle, float* _score, int _Tpre )
{
  // IMPORTANT: caller should assure the validity of _Tpre

  // initialize 
  for (int k = 0; k < K_; ++k) 
    *(_score+k) = 0;

  // sum of tree
  float v = float(param_.v);
  vector<float> s(K_);
  for (int t = Tpre_beg_; t < _Tpre; ++t ) {
    trees_[t].predict (_sapmle, &s[0]);

    for (int k = 0; k < K_; ++k) {
      *(_score+k) += (v*s[k]);
    }
  }

}
int ConsVTLogitBoost::get_class_count()
{
  return K_;
}

int ConsVTLogitBoost::get_num_iter()
{
  return NumIter_;
}

double ConsVTLogitBoost::get_train_loss()
{
  if (NumIter_<1) return DBL_MAX;
  return L_iter_.at<double>(NumIter_-1);
}

void ConsVTLogitBoost::train_init( MLData* _data )
{
  // class count
  K_ = _data->get_class_count();
  // sample count
  N_ = _data->X.rows;

  // F, p
  F_.create(N_,K_); 
  F_ = 0;
  p_.create(N_,K_); 
  calc_p();

  // Loss
  L_.create(N_,1);
  calc_loss(_data);
  L_iter_.create(param_.T,1);

  // iteration for training
  NumIter_ = 0;
  
  // AOTOData
  aotodata_.data_cls_ = _data;
  aotodata_.F_ = &F_;
  aotodata_.p_ = &p_;
  aotodata_.L_ = &L_;

  // trees
  trees_.clear();
  trees_.resize(param_.T);
  for (int t = 0; t < param_.T; ++t) {
    trees_[t].param_.max_leaves_ = param_.J;
    trees_[t].param_.node_size_ = param_.ns;
  }

  // gradient/delta
  abs_grad_.clear();
  abs_grad_.resize(param_.T);

  // for prediction
  Tpre_beg_ = 0;
}

void ConsVTLogitBoost::update_F( int t )
{
  int N = aotodata_.data_cls_->X.rows;
  double v = param_.v;
  vector<float> f(K_);
  for (int i = 0; i < N; ++i) {
    float *psample = aotodata_.data_cls_->X.ptr<float>(i);
    trees_[t].predict(psample,&f[0]);

    double* pF = F_.ptr<double>(i);
    for (int k = 0; k < K_; ++k) {
      *(pF+k) += (v*f[k]);
      // MAX cap
      if ( *(pF+k) > MAXF ) *(pF+k) = MAXF;
    } // for k
  } // for i
}

void ConsVTLogitBoost::calc_p()
{
  int N = F_.rows;
  int K = K_;
  std::vector<double> tmpExpF(K);
  for (int n = 0; n < N; ++n) {
    double tmpSumExpF = 0;
    double* ptrF = F_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      double Fnk = *(ptrF + k);
      double tmp = exp(Fnk);
      tmpExpF[k] = tmp;
      tmpSumExpF += tmp;
    } // for k

    double* ptrp = p_.ptr<double>(n);
    for (int k = 0; k < K; ++k) {
      // TODO: does it make any sense??
      if (tmpSumExpF==0) tmpSumExpF = 1;
      *(ptrp + k) = double( tmpExpF[k]/tmpSumExpF );
    } // for k
  }// for n  
}

void ConsVTLogitBoost::calc_loss( MLData* _data )
{
  const double PMIN = 0.0001;
  for (int i = 0; i < N_; ++i) {
    int yi = int( _data->Y.at<float>(i) );
    double* ptr = p_.ptr<double>(i);
    double pik = *(ptr + yi);

    if (pik<PMIN) pik = PMIN;
    L_.at<double>(i) = (-log(pik));
  }
}

void ConsVTLogitBoost::calc_loss_iter( int t )
{
  double sum = 0;
  int N = L_.rows;
  for (int i = 0; i < N; ++i) 
    sum += L_.at<double>(i);

  L_iter_.at<double>(t) = sum;
}

bool ConsVTLogitBoost::should_stop( int t )
{
  double loss = L_iter_.at<double>(t);
  return ( (loss<EPS_LOSS) ? true : false );
}


void ConsVTLogitBoost::calc_grad( int t )
{
	int N = F_.rows;
	double delta = 0;

	for (int i = 0; i < N; ++i) {
		double* ptr_pi = p_.ptr<double>(i);
		int yi = int( aotodata_.data_cls_->Y.at<float>(i) );

		for (int k = 0; k < K_; ++k) {
			double pik = *(ptr_pi+k);
			if (yi==k) delta += std::abs( 1-pik );
			else       delta += std::abs( -pik );    
		}
	}

	abs_grad_[t] = delta;  
}

void ConsVTLogitBoost::convertToStorTrees()
{
	//initailize
	stor_Trees_.resize(NumIter_);
    
	int i;
	int n = 2*param_.J -1;
    for (i=0; i< NumIter_; i++){
			stor_Trees_[i].nodes_.create(n, 5);// leftID, rightID, splitID, leafID
			stor_Trees_[i].nodes_ = -1;
			stor_Trees_[i].splits_.create(param_.J-1, 2);
			stor_Trees_[i].splits_ = -1;
			stor_Trees_[i].leaves_.create(param_.J, 3);
			stor_Trees_[i].leaves_ = -1;
		}//for

		
	//convert
	for (i=0; i<NumIter_; i++)
	{
		int _leafId = 0;
		int _splitId = 0;
		ConsVTNode* root_Node = &( trees_[i].nodes_.front());
		convert(root_Node, stor_Trees_[i], _leafId, _splitId);
	}
}

void ConsVTLogitBoost::convert(ConsVTNode * _root_Node, 
	StorTree& _sta_Tree, int& _leafId, int& _splitId)
{
	//convert root node
	int nodeId = _root_Node->id_;
	if (_root_Node->parent_ == NULL)
		_sta_Tree.nodes_.at<int>(nodeId, 0)  = -1;
	else _sta_Tree.nodes_.at<int>(nodeId, 0)  = _root_Node->parent_->id_;


	if (_root_Node->left_ == NULL){ //leaf
		_sta_Tree.nodes_.at<int>(nodeId, 4) = _leafId;
		//_sta_Tree.leaves_.at<double>(_leafId, 0) = _root_Node->cls1_;
		//_sta_Tree.leaves_.at<double>(_leafId, 1) = _root_Node->cls2_;
		//_sta_Tree.leaves_.at<double>(_leafId, 2) = _root_Node->fit_val_;
		_leafId ++;
		return ;
	}
	else{//internal node
		_sta_Tree.nodes_.at<int>(nodeId, 1)  = _root_Node->left_->id_;
		_sta_Tree.nodes_.at<int>(nodeId, 2)  = _root_Node->right_->id_;
		_sta_Tree.nodes_.at<int>(nodeId, 3) = _splitId;
		_sta_Tree.splits_.at<double>(_splitId, 0) = _root_Node->split_.var_idx_;
		_sta_Tree.splits_.at<double>(_splitId, 1) = _root_Node->split_.threshold_;
		_splitId ++;
		
		//convert left subtree
		convert(_root_Node->left_, _sta_Tree, _leafId, _splitId);

		//convert right subtree
		convert(_root_Node->right_, _sta_Tree, _leafId, _splitId);
	}
}