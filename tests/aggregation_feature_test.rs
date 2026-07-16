use linal::core::value::Value;
use linal::engine::TensorDb;
use linal::query::logical::{AggregateFunction, Expr, LogicalPlan};
use linal::query::planner::Planner;

#[test]
fn test_aggregation_execution_workflow() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET sales COLUMNS (region: String, amount: Int)
    INSERT INTO sales VALUES ("North", 100)
    INSERT INTO sales VALUES ("South", 200)
    INSERT INTO sales VALUES ("North", 50)
    INSERT INTO sales VALUES ("South", 150)
    INSERT INTO sales VALUES ("East", 300)
    "#;
    linal::dsl::execute_script(&mut db, script).expect("Setup failed");

    let dataset = db.get_dataset("sales").expect("Dataset not found");
    let schema = dataset.schema.clone();

    // Plan: Scan -> Aggregate(Group: region, Agg: Sum(amount))
    let scan = LogicalPlan::Scan {
        dataset_name: "sales".to_string(),
        schema: schema.clone(),
    };

    let agg_plan = LogicalPlan::Aggregate {
        input: Box::new(scan),
        group_expr: vec![Expr::Column("region".to_string())],
        aggr_expr: vec![Expr::AggregateExpr {
            func: AggregateFunction::Sum,
            expr: Box::new(Expr::Column("amount".to_string())),
            alias: None,
        }],
    };

    let planner = Planner::new(&db);
    let physical_plan = planner
        .create_physical_plan(&agg_plan)
        .expect("Plan creation failed");

    let results = physical_plan.execute(&db).expect("Execution failed");

    // Results should have 3 rows (North, South, East)
    assert_eq!(results.len(), 3);

    // Helper to find a row
    let find_row = |region: &str| {
        results
            .iter()
            .find(|r| r.values[0] == Value::String(region.to_string()))
    };

    let north = find_row("North").expect("North not found");
    // amount index is 1 (after region group key)
    assert_eq!(north.values[1], Value::Int(150)); // 100 + 50

    let south = find_row("South").expect("South not found");
    assert_eq!(south.values[1], Value::Int(350)); // 200 + 150

    let east = find_row("East").expect("East not found");
    assert_eq!(east.values[1], Value::Int(300));
}

#[test]
fn test_dsl_aggregation_workflow() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET sales COLUMNS (region: String, amount: Int)
    INSERT INTO sales VALUES ("North", 100)
    INSERT INTO sales VALUES ("South", 200)
    INSERT INTO sales VALUES ("North", 50)
    INSERT INTO sales VALUES ("South", 150)
    INSERT INTO sales VALUES ("East", 300)
    "#;
    linal::dsl::execute_script(&mut db, script).expect("Setup failed");

    let aggr_query = r#"DATASET res FROM sales GROUP BY region SELECT region, SUM(amount)"#;
    linal::dsl::execute_line(&mut db, aggr_query, 0).expect("Aggregation query failed");

    // Check results
    let dataset = db.get_dataset("res").expect("Result dataset not found");
    // Verify rows (North 150, South 350, East 300)

    let find_row = |region: &str| {
        dataset
            .rows
            .iter()
            .find(|r| r.values[0] == Value::String(region.to_string()))
    };

    let north = find_row("North").expect("North not found");
    // The aggregate column name will be "Sum" automatically
    assert_eq!(north.values[1], Value::Int(150));

    let south = find_row("South").expect("South not found");
    assert_eq!(south.values[1], Value::Int(350));

    // Test Global Agg
    let global_query = r#"DATASET global_res FROM sales SELECT COUNT(*)"#;
    linal::dsl::execute_line(&mut db, global_query, 0).expect("Global aggregation failed");
    let global = db.get_dataset("global_res").expect("Global res not found");
    assert_eq!(global.rows.len(), 1);
    assert_eq!(global.rows[0].values[0], Value::Int(5));
}

#[test]
fn test_vector_aggregation() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET vectors COLUMNS (id: Int, v: Vector(2))
    INSERT INTO vectors VALUES (1, [1.0, 2.0])
    INSERT INTO vectors VALUES (2, [3.0, 4.0])
    INSERT INTO vectors VALUES (3, [5.0, 6.0])
    "#;
    linal::dsl::execute_script(&mut db, script).expect("Setup failed");

    // SUM([1,2], [3,4], [5,6]) = [9, 12]
    let sum_query = r#"DATASET v_sum FROM vectors SELECT SUM(v)"#;
    linal::dsl::execute_line(&mut db, sum_query, 0).expect("Vector Sum failed");

    let res = db.get_dataset("v_sum").expect("v_sum not found");
    let row = &res.rows[0];
    if let Value::Vector(v) = &row.values[0] {
        assert_eq!(v, &vec![9.0, 12.0]);
    } else {
        panic!("Expected vector result");
    }

    // MAX([1,2], [3,4], [5,6]) = [5, 6] (Element wise max)
    let max_query = r#"DATASET v_max FROM vectors SELECT MAX(v)"#;
    linal::dsl::execute_line(&mut db, max_query, 0).expect("Vector Max failed");

    let res_max = db.get_dataset("v_max").expect("v_max not found");
    let row_max = &res_max.rows[0];
    if let Value::Vector(v) = &row_max.values[0] {
        assert_eq!(v, &vec![5.0, 6.0]);
    } else {
        panic!("Expected vector result max");
    }
}

#[test]
fn test_computed_column_aggregation() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET items COLUMNS (id: Int, price: Int, qty: Int)
    INSERT INTO items VALUES (1, 10, 2)
    INSERT INTO items VALUES (2, 5, 4)
    INSERT INTO items VALUES (3, 20, 1)
    "#;
    linal::dsl::execute_script(&mut db, script).expect("Setup failed");

    // SUM(price * qty) -> 10*2 + 5*4 + 20*1 = 20 + 20 + 20 = 60
    let query = r#"DATASET total FROM items SELECT SUM(price * qty)"#;
    linal::dsl::execute_line(&mut db, query, 0).expect("Computed sum failed");

    let res = db.get_dataset("total").expect("total not found");
    assert_eq!(res.rows[0].values[0], Value::Int(60));

    // Test precedence: SUM(price + qty * 2)
    // 1: 10 + 2*2 = 14
    // 2: 5 + 4*2 = 13
    // 3: 20 + 1*2 = 22
    // Sum = 14+13+22 = 49
    let query2 = r#"DATASET complex FROM items SELECT SUM(price + qty * 2)"#;
    linal::dsl::execute_line(&mut db, query2, 0).expect("Complex sum failed");

    let res2 = db.get_dataset("complex").expect("complex not found");
    assert_eq!(res2.rows[0].values[0], Value::Int(49));
}

#[test]
fn test_matrix_aggregation() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET matrices COLUMNS (id: Int, val: Matrix(2, 2))
    INSERT INTO matrices VALUES (1, [[1.0, 2.0], [3.0, 4.0]])
    INSERT INTO matrices VALUES (2, [[10.0, 20.0], [30.0, 40.0]])
    "#;
    linal::dsl::execute_script(&mut db, script).expect("Setup failed");

    // SUM
    // [[1+10, 2+20], [3+30, 4+40]] = [[11, 22], [33, 44]]
    let query = r#"DATASET mat_sum FROM matrices SELECT SUM(val)"#;
    linal::dsl::execute_line(&mut db, query, 0).expect("Sum failed");

    let res = db.get_dataset("mat_sum").expect("mat_sum not found");
    if let Value::Matrix(m) = &res.rows[0].values[0] {
        assert_eq!(m.len(), 2);
        assert_eq!(m[0], vec![11.0, 22.0]);
        assert_eq!(m[1], vec![33.0, 44.0]);
    } else {
        panic!("Expected Matrix result, got {:?}", res.rows[0].values[0]);
    }

    // Computed: val * 2
    // [[2, 4], [6, 8]] + [[20, 40], [60, 80]] = [[22, 44], [66, 88]]
    let query2 = r#"DATASET mat_computed FROM matrices SELECT SUM(val * 2)"#;
    linal::dsl::execute_line(&mut db, query2, 0).expect("Computed sum failed");

    let res2 = db
        .get_dataset("mat_computed")
        .expect("mat_computed not found");
    if let Value::Matrix(m) = &res2.rows[0].values[0] {
        assert_eq!(m[0], vec![22.0, 44.0]);
    } else {
        panic!("Expected Matrix result for computed");
    }
}
