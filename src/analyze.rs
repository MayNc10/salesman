// from http://cowlet.org/2016/08/23/linear-regression-in-rust.html

use std::{fs::File, io::Write, vec};

use la::{Matrix, SVD};
use gnuplot::{Figure, AxesCommon, Caption, LineWidth, AutoOption};
use linregress::{RegressionDataBuilder, FormulaRegressionBuilder};
use itertools::{Itertools, TupleWindows};
use petgraph::{graph::{NodeIndex, Node}, Graph, Undirected};

use crate::{layout::Layout, genevo::sim, brute_force::solve};



pub fn rand_layout(node_count: usize, dis_max: u32) -> Layout {
    let mut g = Graph::<(), u32, Undirected>::new_undirected();
    let mut nodes = Vec::new();
    for _ in 0..node_count {
        nodes.push(g.add_node(()));
    }
    for v in nodes.iter().permutations(2) {
        g.add_edge(*v[0], *v[1], rand::random::<u32>() % dis_max + 1);
    }
    Layout { graph: g, known_solution: nodes, best_solution: None}
}

type Solver = fn(Layout, usize) -> u32;

pub fn create_data(layout_generator: fn(usize) -> Layout, 
    solver_to_compare: Solver, baseline: Solver,
    size_range: (usize, usize), runs_per_size: usize, pop_size: usize) -> (Vec<f64>, Vec<f64>)
{
    let mut x = Vec::new();
    let mut y = Vec::new();
    for size in size_range.0..=size_range.1 {
        println!("Simulating size {size}");
        for _ in 0..runs_per_size {
            let layout = layout_generator(size);
            let expected = baseline(layout.clone(), pop_size) as f64;
            let actual = solver_to_compare(layout, pop_size) as f64;
            x.push(size as f64);
            y.push((expected - actual) * 100.0 / expected);
        }
        if size != size_range.0 { plot(x.clone(), y.clone(), "data.csv"); }
    }
    (x, y)
}

fn generate_x_matrix(xs: &Matrix<f64>, order: usize) -> Matrix<f64> {
    let gen_row = {|x: &f64| (0..(order+1)).map(|i| x.powi(i as i32)).collect::<Vec<_>>() };
    let mdata = xs.get_data().iter().fold(vec![], |mut v, x| {v.extend(gen_row(x)); v} );
    Matrix::new(xs.rows(), order+1, mdata)
}

/* 
pub fn linear_regression(xs: &Matrix<f64>, ys: &Matrix<f64>) -> Matrix<f64> {
    let svd = SVD::new(&xs);
    let order = xs.cols()-1;

    let u = svd.get_u();
    // cut down s matrix to the expected number of rows given order (one coefficient per x)
    let s_hat = svd.get_s().filter_rows(&|_, row| { row <= order });
    let v = svd.get_v();

    let alpha = u.t() * ys;
    // "divide each alpha_j by its corresponding s_j"
    // But they are different dimensions, so manually divide each
    // alpha_j by the diagnonal s_j
    let mut mdata = vec![];
    for i in 0..(order+1) {
        mdata.push(alpha.get(i, 0) / s_hat.get(i, i));
    }
    let sinv_alpha = Matrix::new(order+1, 1, mdata);

    v * sinv_alpha
}
*/

static ORDER: usize = 1;

pub fn plot(x: Vec<f64>, y: Vec<f64>, name: &str) {
    let max_x = x.iter().map(|f| f.round() as usize).max().unwrap();

    let data = vec![("Y", y.clone()), ("X1", x.clone())];
    let data = RegressionDataBuilder::new().build_from(data).unwrap();
    let formula = "Y ~ X1";
    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .formula(formula)
        .fit().unwrap();

    let mut x_steps = vec![];
    for i in 0..max_x {
        for j in 0..10 {
            x_steps.push((i as f64) + 0.1 * (j as f64));
        }
    }
    let y_steps = model.predict(vec![("X1", x_steps.clone())]).unwrap();


    let mut f = File::create(name).unwrap();
    for idx in 0..y_steps.len() {
        f.write(format!("{}, {}\n", x_steps[idx], y_steps[idx]).as_bytes()).unwrap();
    }
    for idx in 0..y.len() {
        f.write(format!("{}, {}\n", x[idx], y[idx]).as_bytes()).unwrap();
    }
}

pub fn find_optimal_params(size: usize, dis_max: u32, sample_size: Option<usize>, pval: Option<usize>, base_values: (usize, u64)) 
    -> (usize, u64) 
{
    let sample_size = sample_size.unwrap_or(100);
    let pval = pval.unwrap_or(5);

    // Find maxes
    let mut pop_size_max = base_values.0;
    let mut max_gen_max = base_values.1;
    loop {
        println!("Pop Size max: {}, Max gen max: {}", pop_size_max, max_gen_max);
        let mut points: Vec<usize> = vec![0; sample_size];
        for point in 0..sample_size {
            let l = rand_layout(size, dis_max);
            let ai_best = sim(l.clone(), pop_size_max, max_gen_max);
            let real_best = solve(l.graph).1;
            if ai_best != real_best { points[point] = 1; }
        }
        if points.iter().sum::<usize>() < pval {
            break;
        }
        pop_size_max *= 2;
        max_gen_max *= 2;
    }

    // Find mins
    let mut pop_size_min = base_values.0;
    let mut max_gen_min = base_values.1;
    loop {
        println!("Pop Size min: {}, Max gen min: {}", pop_size_min, max_gen_min);
        let mut points: Vec<usize> = vec![0; sample_size];
        for point in 0..sample_size {
            let l = rand_layout(size, dis_max);
            let ai_best = sim(l.clone(), pop_size_min, max_gen_min);
            let real_best = solve(l.graph).1;
            if ai_best != real_best { points[point] = 1; }
        }
        if points.iter().sum::<usize>() >= pval {
            break;
        }
        pop_size_min /= 2;
        max_gen_min /= 2;
    }    

    loop {
        // We've closed the gap between, return
        if max_gen_max == max_gen_min + 1 && pop_size_max == pop_size_min + 1 {
            break;
        }

        let pop_size = (pop_size_min + pop_size_max) / 2;
        let max_gen = (max_gen_min + max_gen_max) / 2;

        println!("Pop size: {}, max gen: {}", pop_size, max_gen);

        let mut points: Vec<usize> = vec![0; sample_size];
        for point in 0..sample_size {
            let l = rand_layout(size, dis_max);
            let ai_best = sim(l.clone(), pop_size, max_gen);
            let real_best = solve(l.graph).1;
            if ai_best != real_best { points[point] = 1; }
        }
        if points.iter().sum::<usize>() < pval {
            // This setup passes, decrease values
            println!("Passed, decreasing");
            pop_size_max = pop_size;
            max_gen_max = max_gen;
        } else {
            // Increase
            println!("Failed, increasing");
            pop_size_min = pop_size;
            max_gen_min = max_gen;
        } 
    }
    (pop_size_max, max_gen_max)
}