class ReasoningNode:
    def __init__(self, content="", role="assistant",parent=None):
        #node vars
        self.content = content
        self.role = role
        self.parent = parent
        self.children = []
        self.depth= 0 if parent is None else parent.depth + 1
        #MCTS+Beam vars
        self.visits=0
        self.value=0.0
        self.total_value=0.0
    def add_child(self, content=""):
        child = ReasoningNode(content, role="assistant", parent=self)
        self.children.append(child)
        return child
    def get_history(self):
        history = []
        curr = self
        while curr:
            history.append({"role":curr.role,"content":curr.content})
            parent = curr.parent
        return list(reversed(history))
