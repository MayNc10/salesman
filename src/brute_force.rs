use crate::layout::Layout;
use itertools::{Itertools, TupleWindows};
use petgraph::{graph::{NodeIndex, Node}, Graph, Undirected};

pub fn solve(g: Graph::<(), u32, Undirected>) -> u32 {
    // gen all possible loops
    let nodes = (0..g.node_count()).map(|x| NodeIndex::from(x as u32));
    let perms = nodes.permutations(g.node_count());
    //let mut best_path = Vec::new();
    let mut best_length = g.edge_weights().sum::<u32>() * 100_u32;
    for path in perms {
        let mut len = 0;
        for idx in 0..path.len() {
            //println!("{:?}", a);
            let next_idx = (idx + 1) % path.len();
            let n1 = path[idx];
            let n2 = path[next_idx];
            //sprintln!("{:?}, {:?}, {}", n1, n2, self.layout.graph.node_count());
            let e = g.edges_connecting(n1, n2).next().unwrap();
            len += e.weight();
        }
        if len < best_length {
            //best_path = path.clone();
            best_length = len;
        }
    }
    return best_length
}