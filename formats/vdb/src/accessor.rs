use crate::{Node, NodeConst, Tree};

struct Accessor<'a, ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: &'a Tree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
}

impl<'a, ROOT: Node> Accessor<'a, ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub fn get(&mut self)
    where
        ROOT: ~const NodeConst,
    {
        let highest_ancestor: usize = todo!();
        //let meta = Tree::<ROOT>::metas();
        if highest_ancestor == ROOT::LEVEL {
            // root node is the common ancestor.
        } else if highest_ancestor == 0 {
            // leaf node is the common ancestor.
        } else {
            // internal node is the common ancestor.
        }
    }
}
