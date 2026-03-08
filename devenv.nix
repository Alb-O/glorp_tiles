{ pkgs, lib, ... }:

let
  orderQuoteCli = pkgs.rustPlatform.buildRustPackage {
    pname = "order-quote-cli";
    version = "0.1.0";
    src = ./.;
    cargoLock.lockFile = ./Cargo.lock;
  };
  appAgentsText = if builtins.pathExists ./AGENTS.md 
    then (builtins.readFile ./AGENTS.md)
    else "";
in
{
  materializer.ownFragments.rust-app = [ appAgentsText ];
  materializer.mergedFragments = lib.mkAfter [ appAgentsText ];

  treefmt.config.programs.mdformat.enable = true;

  env = {
    SERVICE_NAME = "order-quote-cli";
  };

  packages = [
    pkgs.jq
  ];

  scripts = {
    base-toolchain.exec = ''
      rustc --version
      cargo --version
    '';

    quote-example.exec = ''
      cargo run -- quote 12500 15 express --fragile | jq .
    '';

    health.exec = ''
      cargo run -- health | jq .
    '';

    packaged-health.exec = ''
      ${orderQuoteCli}/bin/order-quote-cli health | jq .
    '';
  };

  outputs.order-quote-cli = orderQuoteCli;

  enterShell = ''
    echo "Run: quote-example"
    echo "Run: health"
    echo "Run: packaged-health"
  '';

  enterTest = ''
    set -euo pipefail

    rustc --version | grep -E "nightly|dev"
    cargo run -- health | jq -e '.status == "ok"'
    cargo run -- quote 12500 15 overnight --fragile | jq -e '.eta_days == 1'
    cargo check --all-targets --all-features
    cargo test --all-targets --all-features
  '';
}
