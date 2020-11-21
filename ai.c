#include <time.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include "ai.h"
#include "utils.h"
#include "priority_queue.h"


struct heap frontier;

float get_reward( node_t* n );
void update_action_score(float latest_score, float best_action_score[], move_t action);
move_t choose_largest_value(float best_action_score[]);
void propagateBackScoreToFirstAction(float best_action_score[], node_t** node, propagation_t propagation );
float calculateAverageChildren(float current_avg, int new_child, float new_acc_reward);


/**
 * Function called by pacman.c
*/
void initialize_ai(){
	heap_init(&frontier);
}

/**
 * function to copy a src into a dst state
*/
void copy_state(state_t* dst, state_t* src){
	//Location of Ghosts and Pacman
	memcpy( dst->Loc, src->Loc, 5*2*sizeof(int) );

    //Direction of Ghosts and Pacman
	memcpy( dst->Dir, src->Dir, 5*2*sizeof(int) );

    //Default location in case Pacman/Ghosts die
	memcpy( dst->StartingPoints, src->StartingPoints, 5*2*sizeof(int) );

    //Check for invincibility
    dst->Invincible = src->Invincible;
    
    //Number of pellets left in level
    dst->Food = src->Food;
    
    //Main level array
	memcpy( dst->Level, src->Level, 29*28*sizeof(int) );

    //What level number are we on?
    dst->LevelNumber = src->LevelNumber;
    
    //Keep track of how many points to give for eating ghosts
    dst->GhostsInARow = src->GhostsInARow;

    //How long left for invincibility
    dst->tleft = src->tleft;

    //Initial points
    dst->Points = src->Points;

    //Remiaining Lives
    dst->Lives = src->Lives;   

}

node_t* create_init_node( state_t* init_state ){
	node_t * new_n = (node_t *) malloc(sizeof(node_t));
	new_n->parent = NULL;	
	new_n->priority = 0;
	new_n->depth = 0;
	new_n->num_childs = 0;
	copy_state(&(new_n->state), init_state);
	new_n->acc_reward =  get_reward( new_n );
	return new_n;
	
}
/**
 * Calculating the heurestic value from the node
*/
float heuristic( node_t* n ){
	float h = 0, i = 0, l = 0, g = 0;
  
    if (n->state.Invincible == 1) { i = 10; }
    if (n->parent != NULL && n->state.Lives < n->parent->state.Lives) { l = 10; }
    if (n->state.Lives < 0) { g = 100; }
    
    h = i - l - g;
	return h;
}
/**
 * Calculating the node accumulated reward for that perticular node
*/
float get_reward ( node_t* n ){
	float reward = 0, h = 0, score = 0, pScore = 0;
    
	h = heuristic(n);
    score = n->state.Points;
    if(n->parent != NULL) { pScore = n->parent->state.Points; }
    
    reward = h + score - pScore;

	float discount = pow(0.99,n->depth);

	return discount * reward;
}


/**
 * Apply an action to node n and return a new node resulting from executing the action
*/
bool applyAction(node_t* n, node_t** new_node, move_t action ){
    bool changed_dir = false;
    
    //Update parents
    (*new_node)->parent = n;
    
    //update with current state
    changed_dir = execute_move_t( &((*new_node)->state), action );	    
        
    //update move
    (*new_node)->move = action;
    
    //update depth
    (*new_node)->depth = n->depth + 1;
    
    //update rewards
    (*new_node)->acc_reward = n->acc_reward + get_reward(*new_node);
    
    //update priority
    (*new_node)->priority = -((*new_node)->depth);    

	return changed_dir;

}

/**
 * Find best action by building all possible paths up to budget
 * and back propagate using either max or avg
 */
move_t get_next_move( state_t init_state, int budget, propagation_t propagation, char* stats, int* currDepth, int* totalGenerated, int* totalExpanded ){

	move_t best_action;

	float best_action_score[4];
	for(unsigned i = 0; i < 4; i++)
	    best_action_score[i] = INT_MIN;

	unsigned generated_nodes = 0;
	unsigned expanded_nodes = 0;
	unsigned max_depth = 0;
	
	//Add the initial node
    node_t* n = create_init_node( &init_state );
	
	//Pushing the first node into the priority queue
	heap_push(&frontier,n);
    
    //Creating an explored array
	node_t** explored_node = (node_t**)malloc(sizeof(node_t*));
    assert(explored_node);
    
    int explored_size = 0;
    
    //Search for each node while the priority queue is not empty
    while (frontier.count > 0){
        // Resizing the explored array size to fit the nodes added
        explored_node = (node_t**)realloc(explored_node,  sizeof(node_t*) * (explored_size + 1));
        assert(explored_node);
            
        //Creating explored node by popping the priority queue nodes
        explored_node[explored_size] = heap_delete(&frontier);
        expanded_nodes++;
        
        max_depth = explored_node[explored_size]->depth;
        //Exploring the nodes as long as the budget exist
        if(explored_size < budget){
            
            //checking for each direction movement
            for(int action = left; action <= down; action++ ){
                node_t* new_node = create_init_node( &init_state );
                
                //Validate if the move to is valid
                bool valid_move = applyAction( explored_node[explored_size] , &new_node, action );
                if(valid_move == false){
                    free(new_node);
                    continue;
                }
                
                //Increment for every valid generated nodes
                generated_nodes++;
                
                //Propagating the new node to the array
                propagateBackScoreToFirstAction(best_action_score, &new_node, propagation);
                
                //Free the new_node if life lost or add the nodes into the queue
                if (explored_node[explored_size]->state.Lives > new_node->state.Lives){
                    free(new_node);
                } else {
                    heap_push(&frontier,new_node);    
                }
                  
            }
            
        } else {
            //Freeing the priority queue array when not needed
            emptyPQ(&frontier);   
        }
        explored_size++;
    }
    
    //Freeing the explored nodes array 
    for (int i = 0; i < explored_size ; i++){
        free(explored_node[i]);
    }
    free(explored_node);

    //choosing the best action with the following function
    best_action = choose_largest_value(best_action_score);
    
	sprintf(stats, "Max Depth: %d Expanded nodes: %d  Generated nodes: %d\n",max_depth,expanded_nodes,generated_nodes);
	 
	if(best_action == left)
		sprintf(stats, "%sSelected action: Left\n",stats);
	if(best_action == right)
		sprintf(stats, "%sSelected action: Right\n",stats);
	if(best_action == up)
		sprintf(stats, "%sSelected action: Up\n",stats);
	if(best_action == down)
		sprintf(stats, "%sSelected action: Down\n",stats);

	sprintf(stats, "%sScore Left %f Right %f Up %f Down %f",stats,best_action_score[left],best_action_score[right],best_action_score[up],best_action_score[down]);
    
    //Updating the depth, total generated nodes, and total expanded nodes across the entire game
    if ((*currDepth) < max_depth){
        (*currDepth) = max_depth;
    }
    (*totalGenerated) += generated_nodes;
    (*totalExpanded) += expanded_nodes;
    
	return best_action;
}
    
/**
 * Update an array if the value given is larger than the current array
 */
void update_action_score(float latest_score, float best_action_score[], move_t action){
    
    
    best_action_score[action] = latest_score;
    
    
}

/**
 * Find best action by selecting the largest score of each action.
 * If same score occur, it will pick random action from each maximum score.
 */
move_t choose_largest_value(float best_action_score[]){
    float max_value = INT_MIN;
    int no_same_max = 0;
    move_t action_array[4];
    
    //searching for each action
    for(int action = left; action <= down; action++ ){
        if(max_value < best_action_score[action]){
            no_same_max = 0;
            max_value = best_action_score[action];
            action_array[no_same_max] = action;
            
        } else if (max_value == best_action_score[action]){
            no_same_max += 1;
            action_array[no_same_max] = action;
        }
    }
    
    if (no_same_max == 0){
        return action_array[0];
    }
    
    int random_number = rand() % (no_same_max + 1);
    return action_array[random_number];
        
}

/**
 * Propagate the values of the current node to the best_action_score array.
 */
void propagateBackScoreToFirstAction(float best_action_score[], node_t** node, propagation_t propagation ){
    node_t** first_node_location = node;
    move_t current_action;
    
    //addressing the location of the first node
    while ( (*first_node_location)->depth > 1 ){                  
        first_node_location = &( (*first_node_location)->parent );
    }
    
    //Updating number of children
    (*first_node_location)->num_childs  = (*first_node_location)->num_childs + 1;
    
    //Updating the current action
    current_action = (*first_node_location)->move;
    
    //getting the current acc_reward for new_node
    float new_node_reward = (*node)->acc_reward;
    
    //propagate the values according to its type
    if (propagation == max){
        if (best_action_score[current_action] < new_node_reward){
            update_action_score(new_node_reward, best_action_score, current_action);
        }
        
    } else {
        float old_score = best_action_score[current_action];
        float current_avg = calculateAverageChildren(old_score, (*first_node_location)->num_childs, new_node_reward);
        
        update_action_score(current_avg, best_action_score, current_action);
    }

}

/**
 * Calculating the average with the new node added
 */
float calculateAverageChildren(float current_avg, int new_child, float new_acc_reward){
    float result;
    
    //old total reward (result will be 0 if this is the first child)
    result = ( current_avg * (new_child-1) ) ;
    
    //Calculating the new average
    result = (result + new_acc_reward) / new_child;

    return result;
}
