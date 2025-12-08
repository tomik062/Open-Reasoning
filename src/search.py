from tree import ReasoningNode
from chat_engine import ChatEngine
import re
import copy
class BeamSearch:
    def __init__(self,engine=None,max_breadth=3,max_depth=10):
        self.root = None
        self.max_breadth = max_breadth
        self.max_depth = max_depth
        if engine:
            self.engine = engine
        else:
            self.engine = ChatEngine()
        self.previous_history=copy.deepcopy(self.engine.history)
    def search(self,question):
        self.root = ReasoningNode(question,'user')
        self.root.value=1
        self.root.total_value=1
        best_nodes=[self.root]
        for depth in range(self.max_depth):
            if not best_nodes:
                return None
            all_step_scores=[]
            for node in best_nodes:
                node_steps=self.expand_logic(node)
                step_scores=self.evaluate_steps(node,node_steps)
                for score,step,parent_node in step_scores:
                    if step.upper()=='SOLVED' and score>=0.8:
                        return node.get_history()
                all_step_scores+=step_scores#add node here

            #Sort by total path value, meaning the current step
            #score (x[0]) + score of path leading to step (x[2].total_value)
            all_step_scores=sorted(all_step_scores,key=lambda x: x[0]+x[2].total_value,reverse=True)
            new_steps=all_step_scores[:self.max_breadth]
            best_nodes=[]
            for score,step,parent_node in new_steps:
                new_node=parent_node.add_child(step)
                new_node.value=score
                new_node.total_value=score+parent_node.total_value
                best_nodes.append(new_node)
        return []




    def expand_logic(self,current_node):
        if current_node.depth>=self.max_depth:
            return []
        new_steps=[]
        responses=0
        if current_node.depth==0:
            context_label = "Problem Statement"
            instruction = "Provide the FIRST logical step to solve this."
        else:
            context_label = "Last Logical Step"
            instruction = "Provide the NEXT logical step."
        user_problem=current_node.get_history()[0]['content']
        base_prompt = (
            f"Problem: {user_problem}\n"
            f"{context_label}: {current_node.content}\n"
            f"Based on the history and the {context_label.lower()}, {instruction}\n"
            "Constraints:\n"
            "1. The output must be a single sentence.\n"
            "2. Output ONLY the step.\n"
            "3. If a solution can be directly derived from previous logical steps in chat history, write it\n"
            "4. If there are no previous logical steps or a solution cant be imminently derived from them,"
            "make a logical step towards a solution.\n"
            "5. If the last step was a solution, output exactly: 'SOLVED'.")
        base_context = (copy.deepcopy(self.previous_history) +
                        copy.deepcopy(current_node.get_history()))
        while responses<self.max_breadth:
            #building a prompt that would create numerous distinct logical steps
            # based on the current state
            prompt = base_prompt
            if new_steps:
                prompt+="\nIMPORTANT: The logical step must be DIFFERENT from these options:\n"
                prompt+="\n".join([f"- {step}" for step in new_steps])
            context=base_context.copy()
            context.append({"role": "user", "content": prompt})
            #generate a logical step from current state
            output=self.engine.generate_answer(context,temperature=1)
            responses+=1
            if output:
                response_text = output['choices'][0]['message']['content'].strip()
                if response_text not in new_steps:
                    new_steps.append(response_text)
        return new_steps

    def evaluate_steps(self,current_node,new_steps):
        user_problem=current_node.get_history()[0]['content']
        step_scores=[]
        base_context = (copy.deepcopy(self.previous_history) +
                        copy.deepcopy(current_node.get_history()))
        for step in new_steps:
            eval_prompt = ("You are a strict logic grader. Rate the last step on a scale of 0.0 to 1.0.\n"
                           "Scoring rules:\n"
                           "1.0: Perfect logic. Essential step.\n"
                           "0.5: Valid but vague or trivial.\n"
                           "0.2: redundant or similar to previous step.\n"
                           "0.0: Logically wrong, impossible according to problem definition, contradicts history, or invalid step.\n"
                           f"Problem: {user_problem}\n"
                           f"Last Logical Step: {step}\n"
                           "Constraints:\n"
                           "1. verify carefully that the logical step is correct, valid, and doesnt result in a contradiction."
                           "with previous logical steps or with the original problem.\n"
                           "2.You must look for contradictions, and score 0 even if there is a chance of contradiction .\n"
                           "3. Penalize repetition heavily.\n"
                           "4. Output ONLY the score as a number, dont include words or letters in your output.\n.")
            context=base_context.copy()
            context.append({"role": "assistant", "content": step})
            context.append({"role": "user", "content": eval_prompt})
            evaluation=self.engine.generate_answer(context,temperature=0.1)
            if evaluation:
                evaluation_text = evaluation['choices'][0]['message']['content'].strip()
                score= self.text_to_score(evaluation_text)
                step_scores.append((score, step, current_node))
            else:
                #failed to evaluate, add default score
                step_scores.append((self.text_to_score(''), step, current_node))
        return step_scores

    def text_to_score(self,text):
        default_score=0.5
        if not text:
            return default_score
        #extracting all numbers from text
        numbers=re.findall(r"(\d+\.?\d*)", text)
        if not numbers:
            return default_score
        #choose the last number, to fit both the default behaviour of a single score output,
        #and a common misbehaviour of "from a scale of 0 to 1 i score this a X" output
        likely_score=float(numbers[-1])
        score= likely_score if 0<=likely_score<=1 else default_score
        return score