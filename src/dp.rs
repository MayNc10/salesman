use std::collections::HashMap;

use crate::layout::Layout;
use itertools::{Itertools, TupleWindows};
use petgraph::{graph::{NodeIndex, Node}, Graph, Undirected};

pub fn solve(g: &Graph::<(), u32, Undirected>) -> u32 {
    let mut map = HashMap::new();
    cost(&g, 0, &mut map, NodeIndex::from(0))
}

fn cost(g: &Graph::<(), u32, Undirected>, nodes_removed: u32, memo: &mut HashMap<(u32, NodeIndex), u32>, end: NodeIndex) -> u32 {
    assert!(g.node_count() <= 32);
    if memo.contains_key(&(nodes_removed, end)) { return *memo.get(&(nodes_removed, end)).unwrap(); }
    
    let mut best_length = g.edge_weights().sum::<u32>() * 100_u32;

    // gen all possible (ignoring final trip back)
    let mut nodes = (0..g.node_count()).map(|x| NodeIndex::from(x as u32))
        .filter(|n| ((1 << n.index()) & nodes_removed) == 0);
    let num_nodes = nodes.clone().count();
    if num_nodes == 1 { // include zero {
        let n = nodes.next().unwrap();
        best_length = *g.edges_connecting(NodeIndex::from(0), n).next().unwrap().weight();
    }
    else {
        //let perms = nodes.permutations(num_nodes);
        for node in nodes.clone().filter(|n| *n != end && n.index() != 0) {
            let len = cost(g, nodes_removed | (1 << end.index()), memo, node) + 
                g.edges_connecting(node, end).next().unwrap().weight();
            if len < best_length { best_length = len; }
        } 
    }   
    memo.insert((nodes_removed, end), best_length);
    best_length
}

fn solve_inner(g: &Graph::<(), u32, Undirected>, nodes_removed: u32, memo: &mut HashMap<u32, u32>, start: usize) -> u32 {
    assert!(g.node_count() <= 32);

    if memo.contains_key(&nodes_removed) { return *memo.get(&nodes_removed).unwrap(); }

    let mut best_length = g.edge_weights().sum::<u32>() * 100_u32;

    // gen all possible (ignoring final trip back)
    let mut nodes = (0..g.node_count()).map(|x| NodeIndex::from(x as u32))
        .filter(|n| ((1 << n.index()) & nodes_removed) == 0);
    let num_nodes = nodes.clone().count();
    if num_nodes == 2 {
        best_length = *g.edges_connecting(nodes.next().unwrap(), nodes.next().unwrap()).next().unwrap().weight();
    }
    else {
        //let perms = nodes.permutations(num_nodes);
        for node in nodes.clone().filter(|n| n.index() != start) {
            for n2 in nodes.clone().filter(|n| *n != node && n.index() != start) {
                let new_nodes_removed = nodes_removed | (1 << n2.index());
                let len = solve_inner(g, new_nodes_removed, memo, node.index()) + 
                    *g.edges_connecting(node, n2).next().unwrap().weight();
                if len < best_length { best_length = len; }
            }
        } 
    }
    memo.insert(nodes_removed, best_length);
    best_length
}