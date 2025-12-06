from tree import ReasoningNode
from chat_engine import ChatEngine
import re
class BeamSearch:
    def __init__(self,engine=None,max_breadth=3,max_depth=20):
        self.root = None
        self.max_breadth = max_breadth
        self.max_depth = max_depth
        if engine:
            self.engine = engine
        else:
            self.engine = ChatEngine()
        self.previous_history=self.engine.history.copy()
    def search(self,question):
        self.root = ReasoningNode(question,'user')
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
        base_prompt = (f"Current State: {current_node.content}\n"
            "Based on this state and the chat history, provide ONE valid, logical next step.\n"
            "Constraints:\n"
            "1. The output must be a single sentence.\n"
            "2. Output ONLY the step.\n"
            "3. If the problem is fully solved, output exactly: 'SOLVED'.")
        base_context = self.previous_history.copy()+current_node.get_history().copy()
        while responses<self.max_breadth:
            #building a prompt that would create numerous distinct logical steps
            # based on the current state
            prompt = base_prompt
            if new_steps:
                prompt+="\nIMPORTANT: The next step must be DIFFERENT from these options:\n"
                prompt+="\n".join([f"- {step}" for step in new_steps])
            context=base_context.copy()
            context.append({"role": "user", "content": prompt})
            #generate a logical step from current state
            output=self.engine.generate_answer(context)
            responses+=1
            if output:
                response_text = output['choices'][0]['message']['content'].strip()
                if response_text not in new_steps:
                    new_steps.append(response_text)
        return new_steps

    def evaluate_steps(self,current_node,new_steps):
        step_scores=[]
        eval_prompt=("Evaluate how logical the last step is on a scale of 0.0 to 1.0, based on chat history.\n"
            "Constraints:\n"
            "1. The output must be a single floating number (e.g., 0.6).\n"
            "2. Be critical: 0.0 is completely illogical, 1.0 is perfect.\n"
            "3. Output ONLY the number, no words or letters.\n"
            "4. Focus your evaluation exclusively on the last step. Do not re-evaluate the previous history,"
            " but judge whether this specific step is a logical continuation of it.")
        base_context = self.previous_history.copy()+current_node.get_history().copy()
        for step in new_steps:
            context=base_context.copy()
            context.append({"role": "assistant", "content": step})
            context.append({"role": "user", "content": eval_prompt})
            evaluation=self.engine.generate_answer(context)
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


