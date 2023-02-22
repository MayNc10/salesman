use std::time::Instant;

use salesman::brute_force::solve;
use salesman::layout::Layout;
use salesman::{layout, brute_force, analyze::*, dp};
use salesman::genevo::sim;

use itertools::{Itertools, TupleWindows};
use petgraph::{graph::{NodeIndex, Node}, Graph, Undirected};

use rand::prelude;



fn main() {
    //let (x, y) = create_data(|size| rand_layout(size, 250), sim, 
    //|g, _| brute_force::solve(g.graph).1, (2, 12), 15, 200);
    
    let layout = rand_layout(10, 50);
    let dp = dp::solve(&layout.graph);
    let brute = solve(layout.graph).1;
    println!("dp: {}, brute-force: {}", dp, brute);


    //let optimal = find_optimal_params(7, 50, None, None, (200, 2000));
    //println!("Optimal population and max generation for size {} are {}, {}", 7, optimal.0, optimal.1);

    //plot(x, y, "data.csv");
}
