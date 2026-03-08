use std::env;
use std::process::ExitCode;

use order_quote_cli::{Command, calculate_quote, parse_command};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct HealthResponse {
    service: String,
    status: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = parse_command(&args)?;

    match command {
        Command::Health => {
            let response = HealthResponse {
                service: env::var("SERVICE_NAME").unwrap_or_else(|_| "order-quote-cli".to_string()),
                status: "ok",
            };
            print_json(&response)?;
            Ok(())
        }
        Command::Quote(request) => {
            let quote = calculate_quote(&request);
            print_json(&quote)?;
            Ok(())
        }
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<(), String> {
    let encoded = serde_json::to_string_pretty(value).map_err(|err| err.to_string())?;
    println!("{encoded}");
    Ok(())
}
