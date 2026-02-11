struct TreeNode[type: Movable]:
    var data: Optional[Self.type]
    var children: List[UnsafePointer[TreeNode[Self.type], MutAnyOrigin]]

    fn __init__(out self, data: Optional[Self.type]):
        self.data = data
        self.children = []

    fn add_child(mut self, child: UnsafePointer[TreeNode[Self.type], MutAnyOrigin]) -> None:
        self.children.append(child)

    fn remove_child(mut self, child: UnsafePointer[TreeNode[Self.type], MutAnyOrigin]) raises -> None:
        index = 0
        for child_iterator in self.children:
            if child_iterator == child:
                _ = self.children.pop(index)
                return
            index += 1
        raise Error('Could not remove tree node child')
