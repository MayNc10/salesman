use std::rc::Rc;
use std::sync::Arc;

use genevo::operator::CrossoverOp;
use genevo::random::{SliceRandom, random_index};
use genevo::{operator::prelude::*, prelude::*, random::Rng, types::fmt::Display};

use petgraph::dot::Dot;
use petgraph::{Graph, Undirected};
use petgraph::graph::NodeIndex;

use crate::layout::Layout;

const POPULATION_SIZE: usize = 200;
const GENERATION_LIMIT: u64 = 2000;
const NUM_INDIVIDUALS_PER_PARENTS: usize = 2;
const SELECTION_RATIO: f64 = 0.7;
const MUTATION_RATE: f64 = 0.5;
const REINSERTION_RATIO: f64 = 0.7;
const HIGHEST_FITNESS: i64 = 10000;

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
struct Path {
    pub path: Vec<NodeIndex>,
}

impl Genotype for Path {
    type Dna = NodeIndex;
}

#[derive(Clone, Debug)]
struct FitnessCalc<'a> {
    layout: &'a Layout,
    highest_fitness: i64,
}
impl<'a> FitnessCalc<'a> {
    pub fn new(layout: &'a Layout, highest_fitness: i64) -> FitnessCalc {
        FitnessCalc { layout, highest_fitness}
    }
}
impl FitnessFunction<Path, i64> for FitnessCalc<'_> {
        fn fitness_of(&self, a: &Path) -> i64 {
        let mut score = 0;
        for idx in 0..a.path.len() {
            //println!("{:?}", a);
            let next_idx = (idx + 1) % a.path.len();
            let n1 = a.path[idx];
            let n2 = a.path[next_idx];
            //sprintln!("{:?}, {:?}, {}", n1, n2, self.layout.graph.node_count());
            let e = self.layout.graph.edges_connecting(n1, n2).next().unwrap();
            score += e.weight();
        }

        (self.highest_fitness - score as i64)
    }
    fn average(&self, a: &[i64]) -> i64 {
        (a.iter().sum::<i64>() as f32 / a.len() as f32 + 0.5).floor() as i64
    }
    fn highest_possible_fitness(&self) -> i64 {
        self.highest_fitness
    }
    fn lowest_possible_fitness(&self) -> i64 {
        0
    }
}

impl BreederGenomeMutation for Path {
    type Dna = NodeIndex;

    fn mutate_genome<R>(
            genome: Self,
            mutation_rate: f64,
            range: &<Self as Genotype>::Dna,
            precision: u8,
            min_value: &<Self as Genotype>::Dna,
            max_value: &<Self as Genotype>::Dna,
            rng: &mut R,
        ) -> Self
        where
            R: Rng + Sized {
        // Just stole this from the implementation on Vec, may break
        let genome_length = genome.path.len();
        let num_mutations =
            ((genome_length as f64 * mutation_rate) + rng.gen::<f64>()).ceil() as usize;
        let mut mutated = genome.clone();
        for _ in 0..num_mutations {
            // Maybe use precision?
            let idx1 = rng.gen::<usize>() % genome.path.len();
            let idx2 = rng.gen::<usize>() % genome.path.len();
            let first_elem = mutated.path[idx1];
            mutated.path.remove(idx1);
            mutated.path.insert(idx2, first_elem);
        }
        assert!({
            let mut v = true;
            for i in 0..mutated.path.len() {
                v &= mutated.path.contains(&NodeIndex::from(i as u32));
            }
            v
        });
        //assert_ne!(genome.path, mutated.path);
        mutated
    }
}

fn find_in_vec(vec: &Vec<NodeIndex>, val: &NodeIndex) -> usize {
    for idx in 0..vec.len() {
        if vec[idx] == *val {
            return idx;
        }
    }
    panic!("Coudln't find {:?} in {:?}", val, vec);
}

impl CrossoverOp<Path> for PartiallyMappedCrossover  {
    fn crossover<R>(&self, parents: genevo::genetic::Parents<Path>, rng: &mut R) -> genevo::genetic::Children<Path>
    where
        R: Rng + Sized 
    {
        if parents.len() > 2 { 
            //println!("More parents than 2, ignoring ...");    
        }
        let mut child = Vec::with_capacity(parents[0].path.len());
        for _ in 0..parents[0].path.len() { child.push(None); }

        // using pmx https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
        let p1 = rng.gen::<usize>() % parents[0].path.len();
        let p2 = rng.gen::<usize>() % (parents[0].path.len() - p1) + p1;

        for idx in p1..p2 { child[idx] = Some(parents[0].path[idx]); }
        
        for idx in p1..p2 {
            if !child.contains(&Some(parents[1].path[idx])) {
                let val = child[idx].unwrap();
                //println!("Val: {:?}, p1: {:?}", val, parents[1].path);
                let new_idx = find_in_vec(&parents[1].path, &val);
                if child[new_idx].is_some() {
                    let k = child[new_idx].unwrap();
                    //println!("K: {:?}, p1: {:?}", k, parents[1].path);
                    let new_idx = find_in_vec(&parents[1].path, &k);
                    child[new_idx] = Some(parents[1].path[idx]);
                }
                else {
                    child[new_idx] = Some(parents[1].path[idx]);
                }
            }
        }
         

        let mut child_idx = 0;
        for gene in &parents[1].path {
            if !child.contains(&Some(*gene)) {
                while child[child_idx].is_some() { child_idx += 1; }
                child[child_idx] = Some(*gene);
            }
        }
        // println!("p1: {:?}, p2: {:?}, child: {:?}", parents[0].path, parents[1].path, child);
        // This is probably wrong
        Vec::from([Path { path: child.into_iter().map(|x| x.unwrap()).collect() }])
    }
}

struct Paths<'a> {
    layout: &'a Layout,
}
impl<'a> Paths<'a> {
    pub fn new(layout: &'a Layout) -> Paths {
        Paths { layout }
    }
}
impl GenomeBuilder<Path> for Paths<'_> {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> Path
    where
        R: Rng + Sized 
    {
        let mut path = self.layout.known_solution.clone();
        path.shuffle(rng);
        Path { path: path.into_iter().map(|node| node).collect() }
    }
}

pub fn sim(layout: Layout, pop_size: usize, gen_limit: u64) -> u32 {
    let initial_pop: Population<Path> = build_population()
        .with_genome_builder(Paths::new(&layout))
        .of_size(pop_size)
        .uniform_at_random();

    let mut salesman_sim = simulate(
        genetic_algorithm()
            .with_evaluation(FitnessCalc::new(&layout, HIGHEST_FITNESS))
            .with_selection(RouletteWheelSelector::new(
                SELECTION_RATIO,
                NUM_INDIVIDUALS_PER_PARENTS
            ))
            .with_crossover(PartiallyMappedCrossover::new())
            .with_mutation(BreederValueMutator::new(
                MUTATION_RATE,
                NodeIndex::from(layout.known_solution.len() as u32 - 1),
                3,
                NodeIndex::from(0),
                NodeIndex::from(layout.known_solution.len() as u32 - 1),
            ))
            .with_reinsertion(ElitistReinserter::new(
                FitnessCalc::new(&layout, HIGHEST_FITNESS),
                false,
                REINSERTION_RATIO,
            ))
            .with_initial_population(initial_pop)
            .build(),
    )
    .until(or(
        FitnessLimit::new(FitnessCalc::new(&layout, HIGHEST_FITNESS).highest_possible_fitness()),
        GenerationLimit::new(gen_limit),
    ))
    .build();

    loop {
        let result = salesman_sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                //println!(
                //    "Step: generation: {}, average fitness: {}, lowest fitness: {} \
                //     best fitness: {}, duration: {}, processing_time: {}",
                //    step.iteration,
                //    evaluated_population.average_fitness(),
                //    evaluated_population.lowest_fitness(),
                //    best_solution.solution.fitness,
                //    step.duration.fmt(),
                //    step.processing_time.fmt()
                //);
            },
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                return (HIGHEST_FITNESS as u32 - best_solution.solution.fitness as u32);
                //println!("{}", stop_reason);
                //println!(
                //    "Final result after {}: generation: {}, \
                //     Best solution {:?}
                //     fitness {} found in generation {}, processing_time: {}",
                //    duration.fmt(),
                //    step.iteration,
                //    best_solution.solution.genome.path,
                //    best_solution.solution.fitness,
                //    best_solution.generation,
                //    processing_time.fmt()
                //);
                //break;
            },
            Err(error) => {
                println!("{}", error);
                break;
            },
        }
    }
    return 0;
    //println!("{:?}", Dot::with_config(&layout.graph, &[]));
}
