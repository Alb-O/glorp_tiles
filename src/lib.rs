#![forbid(unsafe_code)]

use std::str::FromStr;

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Standard,
    Express,
    Overnight,
}

impl FromStr for Priority {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "express" => Ok(Self::Express),
            "overnight" => Ok(Self::Overnight),
            _ => Err(format!(
                "invalid priority '{value}', expected standard|express|overnight"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteRequest {
    pub subtotal_cents: u64,
    pub distance_km: u32,
    pub priority: Priority,
    pub fragile: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QuoteResponse {
    pub shipping_cents: u64,
    pub handling_cents: u64,
    pub total_cents: u64,
    pub eta_days: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Health,
    Quote(QuoteRequest),
}

pub fn parse_command(args: &[String]) -> Result<Command, String> {
    match args.first().map(String::as_str) {
        Some("health") => Ok(Command::Health),
        Some("quote") => parse_quote(args).map(Command::Quote),
        _ => Err(usage()),
    }
}

pub fn calculate_quote(req: &QuoteRequest) -> QuoteResponse {
    let base = 399_u64;
    let distance = u64::from(req.distance_km) * 12;
    let subtotal_surcharge = if req.subtotal_cents < 5000 { 175 } else { 0 };
    let fragile_fee = if req.fragile { 250 } else { 0 };
    let priority_fee = match req.priority {
        Priority::Standard => 0,
        Priority::Express => 550,
        Priority::Overnight => 1200,
    };
    let shipping_cents = base + distance + subtotal_surcharge + fragile_fee + priority_fee;
    let handling_cents = req.subtotal_cents.div_ceil(1000) * 35;
    let total_cents = req.subtotal_cents + shipping_cents + handling_cents;
    let eta_days = match req.priority {
        Priority::Standard => 5,
        Priority::Express => 2,
        Priority::Overnight => 1,
    };

    QuoteResponse {
        shipping_cents,
        handling_cents,
        total_cents,
        eta_days,
    }
}

pub fn usage() -> String {
    "usage:\n  order-quote-cli health\n  order-quote-cli quote <subtotal_cents> <distance_km> <standard|express|overnight> [--fragile]".to_string()
}

fn parse_quote(args: &[String]) -> Result<QuoteRequest, String> {
    if args.len() < 4 {
        return Err(usage());
    }

    let subtotal_cents = args[1]
        .parse::<u64>()
        .map_err(|_| "subtotal_cents must be an integer".to_string())?;
    let distance_km = args[2]
        .parse::<u32>()
        .map_err(|_| "distance_km must be an integer".to_string())?;
    let priority = Priority::from_str(&args[3])?;
    let fragile = args.iter().skip(4).any(|arg| arg == "--fragile");

    Ok(QuoteRequest {
        subtotal_cents,
        distance_km,
        priority,
        fragile,
    })
}

#[cfg(test)]
mod tests {
    use super::{Command, Priority, QuoteRequest, calculate_quote, parse_command};

    #[test]
    fn quote_defaults_work() {
        let req = QuoteRequest {
            subtotal_cents: 10_000,
            distance_km: 20,
            priority: Priority::Standard,
            fragile: false,
        };
        let quote = calculate_quote(&req);
        assert_eq!(quote.shipping_cents, 639);
        assert_eq!(quote.handling_cents, 350);
        assert_eq!(quote.total_cents, 10_989);
        assert_eq!(quote.eta_days, 5);
    }

    #[test]
    fn quote_express_fragile_adds_fees() {
        let req = QuoteRequest {
            subtotal_cents: 4_500,
            distance_km: 10,
            priority: Priority::Express,
            fragile: true,
        };
        let quote = calculate_quote(&req);
        assert_eq!(quote.shipping_cents, 1_494);
        assert_eq!(quote.handling_cents, 175);
        assert_eq!(quote.total_cents, 6_169);
        assert_eq!(quote.eta_days, 2);
    }

    #[test]
    fn parse_quote_command() {
        let args = vec![
            "quote".to_string(),
            "12500".to_string(),
            "15".to_string(),
            "overnight".to_string(),
            "--fragile".to_string(),
        ];
        let command = parse_command(&args).expect("quote command should parse");
        let Command::Quote(req) = command else {
            panic!("expected quote command");
        };

        assert_eq!(req.subtotal_cents, 12_500);
        assert_eq!(req.distance_km, 15);
        assert_eq!(req.priority, Priority::Overnight);
        assert!(req.fragile);
    }
}
