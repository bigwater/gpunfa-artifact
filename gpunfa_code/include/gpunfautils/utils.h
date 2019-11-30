/*
 * utils.h
 *
 *  Created on: May 16, 2018
 *      Author: hyliu
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "commons/NFA.h"
#include <string>
#include <vector>
#include "array2.h"
#include "common.h"
#include "commons/SymbolStream.h"
#include <map>
#include "commons/compatible_group_helper.h"


using std::vector;
using std::string;
using std::pair;
using std::map;


namespace nfa_utils {

	//only for debug
	NFA *focus_on_one_cc(NFA* bignfa, string str_id);

	NFA *cut_by_normalized_depth(NFA *nfa, double normalized_depth_limit);

	void compress_alphabet(NFA *, map<uint8_t, uint8_t> &alphabet_mapping);

	void print_cc(NFA *nfa, int cc_id);

	/* whoever calls, whoever deletes */
	Array2<int4> *create_int4_tt_for_nfa(NFA *nfa);

	Array2<STE_dev<4> > *create_list_of_STE_dev(NFA *nfa);

	// compress edge
	Array2<STE_dev4 > *create_STE_dev4_compressed_edges(NFA *nfa);


	Array2<unsigned long long> *create_ull64_for_nfa(NFA *nfa);


	Array2<STE_dev4_compressed_matchset > *create_STE_dev4_compressed_edges_and_compressed_matchset(NFA *nfa); 

	Array2<STE_dev4_compressed_matchset_allcomplete > *create_STE_dev4_compressed_edges_and_compressed_matchset_allcomplete(NFA *nfa);


	// begin new implementation for matchset compression 20190121
	Array2<STE_nodeinfo_new_imp> *create_STE_nodeinfos_new(NFA *cc);
    Array2<STE_nodeinfo_new_imp2> *create_STE_nodeinfos_new2(NFA *cc);
	Array2<STE_matchset_new_imp> *create_STE_matchset_new (NFA *cc);
	// end new implementation for matchset compression 20190121


	// 20190202
	Array2<STE_nodeinfo_new_imp_withcg> *create_STE_nodeinfos_new_withcg(NFA *cc);
	// 

	Array2<char> *get_attribute_array(NFA *); 

	template <class T>
	Array2<T> *create_nodelist_for_nfa_groups(vector<NFA*> vecs, int block_size, Array2<T> * (*func1)(NFA* )) {
		if (vecs.size() == 0) {
			return NULL;
		}

		Array2<T> *res = new Array2<T> (vecs.size() * block_size);

		for (int i = 0; i < vecs.size(); i++) {
			auto nfa = vecs[i];
			auto cur_array2 = (*func1)(nfa);

			for (int j = 0; j < cur_array2->size(); j++) {
				res->set(i * block_size + j, cur_array2->get(j));
			}

			delete cur_array2;
		}

		return res;
	}

	template <class T>
	Array2<T> *create_nodelist_for_nfa_groups2(vector<NFA*> vecs, Array2<T> * (*func1)(NFA* )) {
		if (vecs.size() == 0) {
			return NULL;
		}

		int total_num_states = 0 ;
		for (auto nn : vecs) {
			total_num_states += nn->size();
		}

		Array2<T> *res = new Array2<T> (total_num_states);

		int tt = 0;
		for (int i = 0; i < vecs.size(); i++) {
			auto nfa = vecs[i];
			auto cur_array2 = (*func1)(nfa);

			for (int j = 0; j < cur_array2->size(); j++) {
				res->set(tt++, cur_array2->get(j));
			}

			delete cur_array2;
		}

		assert(tt == total_num_states);

		return res;
	}


    template <class T>
    Array2<T> *create_nodelist(vector<NFA*> vecs, Array2<T> * (*func1)(NFA* ), int total_size) {
        if (vecs.size() == 0) {
            return NULL;
        }

        Array2<T> *res = new Array2<T> (total_size);

        int tt = 0;
        for (int i = 0; i < vecs.size(); i++) {
            auto nfa = vecs[i];
            auto cur_array2 = (*func1)(nfa);

            for (int j = 0; j < cur_array2->size(); j++) {
                res->set(tt++, cur_array2->get(j));
            }

            delete cur_array2;
        }

        return res;
    }



    template <class T>
	Array2<T> **create_nodelist_independent_cc(vector<NFA*> vecs, Array2<T> * (*func1)(NFA* )) {
		assert(vecs.size() > 0);

		Array2<T> **res = new Array2<T> *  [vecs.size()];

		for (int i = 0; i < vecs.size(); i++) {
			auto nfa = vecs[i];
			auto cur_array2 = (*func1)(nfa);

			res[i] = cur_array2;

		}

		return res;
	}

	set<uint8_t> get_alphabet_from_nfa(NFA *nfa);


	// for active+active maching process. 
	void add_fake_start_node_for_ccs(vector<NFA *> ccs);

	Array2<uint8_t> *get_array2_for_input_stream(const SymbolStream& symbolstream);

	Array2<int> *get_nfa_size_array2(const vector<NFA*> &nfas);

	void test_compress_matchset1(NFA *nfa);

	bool is_all_complete(NFA *nfa);

	int get_start_node_id(NFA *nfa);

	Array2<int> *get_allways_enabled_start_node_ids(NFA *nfa);

	NFA *filter_and_create_new_NFA(NFA *nfa, const std::set<string> & states_to_keep);
	

	

	vector<NFA *> order_nfa_intid_by_hotcold(const vector<NFA *> &grouped_nfas); 

	

	
	void print_indegrees(const vector<NFA *> nfas);

	

}



namespace bitvec {
	void set_bit(int *arr, int len_arr, int k_bit); 
}





class HotColdHelper {

	map<string, int> freq_map;

	int threshold;


public:
	HotColdHelper();
	void set_threshold(int threshold);
	bool is_hot(string state_id) const ;
	bool is_cold(string state_id) const ;
};




#endif /* UTILS_H_ */



