from testing import TestSuite, assert_equal, assert_raises

from src.data_types import TreeNode

fn test_tree_node() raises:
    comptime ChildrenDataPair = Tuple[UnsafePointer[TreeNode[Int], MutAnyOrigin], List[UnsafePointer[TreeNode[Int], MutAnyOrigin]]]

    root_node = TreeNode[Int](0)
    root_node_ptr = UnsafePointer(to=root_node)
    node1 = TreeNode[Int](1)
    node1_ptr = UnsafePointer(to=node1)
    node2 = TreeNode[Int](2)
    node2_ptr = UnsafePointer(to=node2)
    node1_node1 = TreeNode[Int](3)
    node1_node1_ptr = UnsafePointer(to=node1_node1)
    external_node = TreeNode[Int](3)
    external_node_ptr = UnsafePointer(to=external_node)

    children_data = [
        ChildrenDataPair(root_node_ptr, [node1_ptr, node2_ptr]),
        ChildrenDataPair(node1_ptr, [node1_node1_ptr])
    ]
    non_children_data = [
        ChildrenDataPair(root_node_ptr, [root_node_ptr, node1_node1_ptr, external_node_ptr]),
        ChildrenDataPair(node1_ptr, [root_node_ptr, node1_ptr, node2_ptr, external_node_ptr]),
        ChildrenDataPair(node2_ptr, [root_node_ptr, node1_ptr, node2_ptr, node1_node1_ptr, external_node_ptr]),
        ChildrenDataPair(external_node_ptr, [root_node_ptr, node1_ptr, node2_ptr, node1_node1_ptr, external_node_ptr])
    ]

    for node, children in children_data:
        for child in children:
            node[].add_child(child)

    for node, children in children_data:
        for child, node_child in zip(children, node[].children):
            assert_equal(child, node_child)
            assert_equal(child[].children, node_child[].children)
            assert_equal(child[].data.value(), node_child[].data.value())

    for node, non_children in non_children_data:
        with assert_raises():
            for non_child in non_children:
                node[].remove_child(non_child)

    for child in children_data[0][1]:
        children_data[0][0][].remove_child(child)
    non_children_data.append(children_data[0])
    _ = children_data.pop(0)

    for node, non_children in non_children_data:
        with assert_raises():
            for non_child in non_children:
                node[].remove_child(non_child)

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
