use core::num;
use std::cmp::min;

use petgraph::{Graph, Undirected};
use petgraph::graph::NodeIndex;
use petgraph::dot;

use probability::distribution::Gaussian;

use probability::prelude::Sample;
use probability::source;
use rand::prelude::*;

pub static DEFAULT_CONN_LENGTH: u32 = 5;
pub static SIGMA_SCALAR: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct Layout {
    pub graph: Graph<(), u32, Undirected>,
    pub known_solution: Vec<NodeIndex>,
    pub best_solution: Option<Vec<NodeIndex>>,
}

impl Layout {
    pub fn new(num_nodes: u32, length: Option<u32>) -> (Layout, Vec<Vec<Option<u32>>>) {
        let mut nodes = Vec::with_capacity(num_nodes as usize);
        let mut g = Graph::<(), u32, Undirected>::new_undirected();
        for _ in 0..num_nodes {
            nodes.push(g.add_node(()));
        }
        let len = length.unwrap_or(DEFAULT_CONN_LENGTH * num_nodes);
        // generate known path
        let mut current_len_left = len;
        let mut nodes_left = nodes.clone();
        let first_node_idx = random::<usize>() % nodes_left.len();
        let first_node = nodes_left[first_node_idx];
        let mut current_node = first_node;
        nodes_left.remove(first_node_idx);

        let mut connection_matrix = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            let mut row = Vec::with_capacity(num_nodes as usize);
            for _ in 0..num_nodes {
                row.push(None);
            }
            connection_matrix.push(row);
        }

        let mut path = Vec::with_capacity(num_nodes as usize);
        let mut source = source::default(random());
        while nodes_left.len() > 0 {
            path.push(current_node);

            // generate gaussian and new len
            let max = min((2.0 * current_len_left as f64 / nodes_left.len() as f64).round() as u32, current_len_left - nodes_left.len() as u32);
            let min = 1;
            let gaussian = Gaussian::new(current_len_left as f64 / nodes_left.len() as f64, (max - min) as f64 * SIGMA_SCALAR);
            let mut chosen_len = gaussian.sample(&mut source).round() as i32;
            //println!("{}, {}", chosen_len, max);
            if chosen_len < min as i32 { chosen_len = min as i32; }
            else if chosen_len > max as i32 { chosen_len %= max as i32; }
            let chosen_len = chosen_len as u32;

            let target_node_idx = random::<usize>() % nodes_left.len();
            let target_node = nodes_left[target_node_idx];
            nodes_left.remove(target_node_idx);
            connection_matrix[current_node.index()][target_node.index()] = Some(chosen_len);
            connection_matrix[target_node.index()][current_node.index()] = Some(chosen_len);

            current_len_left -= chosen_len;
            current_node = target_node;

            //println!("{:#?}", dot::Dot::with_config(&g, &[]));
        }
        path.push(current_node);
        connection_matrix[current_node.index()][first_node.index()] = Some(current_len_left);
        connection_matrix[first_node.index()][current_node.index()] = Some(current_len_left);

        let mut rng = rand::thread_rng();

        let gaussian = Gaussian::new(
            connection_matrix.iter().fold(0, |acc, x| acc + x.iter()
                .fold(0, |acc, num| acc + num.unwrap_or(0))) as f64 
                / num_nodes as f64, 
            (
                connection_matrix.iter()
                    .map(|x| x.iter()
                    .filter(|num| num.is_some())
                    .max().unwrap()).max().unwrap().unwrap() - 
                connection_matrix.iter()
                    .map(|x| x.iter()
                    .filter(|num| num.is_some())
                    .min().unwrap()).min().unwrap().unwrap()) 
                as f64 * SIGMA_SCALAR);
        // make sure every node is connected to all the others
        for row in 0..num_nodes {
            for col in 0..num_nodes {
                if connection_matrix[row as usize][col as usize].is_none() {
                    let len = gaussian.sample(&mut source).abs().round() as u32;
                    connection_matrix[row as usize][col as usize] = Some(len);
                    connection_matrix[col as usize][row as usize] = Some(len);
                }
            }
        }

        // now that we have the adjecency matrix, use it to generate connections
        for row in 0..num_nodes {
            for col in 0..row {
                g.add_edge(NodeIndex::from(row), NodeIndex::from(col), connection_matrix[row as usize][col as usize].unwrap());
            }
        }

        //println!("{:#?}", dot::Dot::with_config(&g, &[]));

        (Layout { graph: g, known_solution: path, best_solution: None }, connection_matrix)
    }
}


