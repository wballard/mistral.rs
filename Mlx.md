# MLX Backend Integration Specification for mistral.rs

## Overview

This specification outlines the integration of Apple's MLX framework into mistral.rs as a new backend, enabling native Apple Silicon optimization while maintaining compatibility with existing mistral.rs features including chat templates, tool callbacks, and MCP support.

## Executive Summary

- **Goal**: Add MLX backend support to mistral.rs for Apple Silicon optimization
- **Approach**: Integrate `mlx-rs` crate as a new backend alongside existing Candle backend
- **Pattern**: Follow existing `GgufModelBuilder` architecture for consistency
- **Target**: Full feature parity with existing backends (chat templates, tools, MCP)
- **Test Models**: 3 small MLX Community models for integration testing

## Technical Architecture

### 1. Backend Integration Structure

```rust
// New MLX backend module structure
src/
├── backends/
│   ├── candle/          // Existing
│   ├── mlx/             // New MLX backend
│   │   ├── mod.rs
│   │   ├── loader.rs
│   │   ├── model.rs
│   │   └── device.rs
│   └── mod.rs
├── pipeline/
│   ├── loaders/
│   │   ├── gguf_loader.rs    // Existing
│   │   └── mlx_loader.rs     // New MLX loader
│   └── normal.rs
└── lib.rs
```

### 2. MLX Model Builder API

Following the `GgufModelBuilder` pattern:

```rust
use mistralrs::{MlxModelBuilder, MlxSpecificConfig};

// Basic usage - mirrors GgufModelBuilder API
let model = MlxModelBuilder::new("mlx-community/gemma-3-1b-it-4bit")
    .with_logging()
    .with_isq(IsqType::Q4K)
    .build()
    .await?;

// Advanced configuration
let model = MlxModelBuilder::new("mlx-community/Llama-3.2-3B-8bit")
    .with_mlx_config(MlxSpecificConfig {
        device_preference: MlxDevice::Auto,
        lazy_evaluation: true,
        unified_memory: true,
    })
    .with_chat_template_literal(custom_template)
    .with_tool_config(tool_config)
    .build()
    .await?;
```

### 3. Core Components

#### A. MLX Model Loader (`src/pipeline/loaders/mlx_loader.rs`)

```rust
pub struct MlxLoader {
    mlx_config: MlxSpecificConfig,
    model_path: String,
}

impl MlxLoader {
    pub async fn load_model_from_hf(
        &self,
        model_id: &str,
    ) -> Result<(mlx_rs::Model, mlx_rs::Tokenizer), Error> {
        // Integration with mlx-rs crate
        // Handle model downloading and conversion
    }
    
    pub fn supports_model(&self, config: &ModelConfig) -> bool {
        // Check if model is MLX compatible
    }
}
```

#### B. MLX Backend Implementation (`src/backends/mlx/mod.rs`)

```rust
use mlx_rs::{Array, Device, Stream};

pub struct MlxBackend {
    device: Device,
    stream: Stream,
}

pub struct MlxModel {
    inner: mlx_rs::Model,
    tokenizer: mlx_rs::Tokenizer,
    backend: MlxBackend,
}

impl Pipeline for MlxModel {
    async fn step(
        &self,
        input_toks: &[u32],
        // ... standard pipeline parameters
    ) -> Result<GenerationResponse, Error> {
        // MLX-specific inference implementation
        // Convert input to MLX arrays
        // Run inference using mlx-rs
        // Convert output back to mistralrs format
    }
}
```

#### C. Device Management (`src/backends/mlx/device.rs`)

```rust
#[derive(Debug, Clone)]
pub enum MlxDevice {
    Cpu,
    Gpu,
    Auto,  // MLX unified memory - let MLX decide
}

pub struct MlxDeviceManager {
    current_device: MlxDevice,
    unified_memory: bool,
}

impl MlxDeviceManager {
    pub fn optimal_device() -> MlxDevice {
        // MLX advantage: unified memory means less device management
        MlxDevice::Auto
    }
}
```

### 4. CLI Integration

Extend mistral.rs CLI to support MLX models:

```bash
# New 'mlx' subcommand following existing pattern
./mistralrs-server mlx \
  -m mlx-community/gemma-3-1b-it-4bit \
  --port 1234 \
  --mlx-device auto \
  --mlx-lazy-eval true

# ISQ support for MLX models
./mistralrs-server --isq Q4K mlx \
  -m mlx-community/Llama-3.2-3B-8bit

# Tool calling support
./mistralrs-server mlx \
  -m mlx-community/gemma-3-1b-it-4bit \
  --enable-tools \
  --mcp-port 3000
```

### 5. Feature Parity Requirements

#### A. Chat Templates
- **Requirement**: Full compatibility with existing chat template system
- **Implementation**: MLX tokenizer integration with mistralrs chat template engine
- **Test**: Verify chat templates work identically across backends

```rust
// Must work identically for MLX backend
let messages = vec![
    ChatMessage::User("Write a Python function".to_string()),
];
let response = model.send_chat_request(messages).await?;
```

#### B. Tool Callbacks
- **Requirement**: Support all existing tool callback functionality
- **Implementation**: MLX backend must integrate with mistralrs tool system
- **Test**: Function calling must work identically

```rust
// Tool calling should work transparently
let tools = vec![
    Tool::new("calculate", calculate_function),
];
let model = MlxModelBuilder::new(model_id)
    .with_tools(tools)
    .build().await?;
```

#### C. MCP Support
- **Requirement**: Full MCP server/client compatibility
- **Implementation**: MLX models participate in MCP protocol seamlessly
- **Test**: MCP examples should work with MLX backend

## Integration Test Suite

### Test Models Selection

Based on mlx-community analysis, these 3 models provide optimal test coverage:

1. **`mlx-community/gemma-3-1b-it-4bit` (1B parameters)**
   - **Size**: ~500MB (smallest viable test model)
   - **Features**: Instruction-tuned, chat template support
   - **Use Case**: Basic functionality testing
   - **Download**: Fast, CI-friendly

2. **`mlx-community/Llama-3.2-3B-8bit` (3B parameters)**
   - **Size**: ~3GB (medium test model)
   - **Features**: Advanced chat capabilities, tool calling
   - **Use Case**: Feature completeness testing
   - **Download**: Reasonable for thorough testing

3. **`mlx-community/stable-code-3b-mlx` (3B parameters)**
   - **Size**: ~3GB (code-focused model)
   - **Features**: Code generation, technical tasks
   - **Use Case**: Specialized functionality testing
   - **Download**: Tests different model architectures

### Test Cases

#### A. Basic Functionality Tests
```rust
#[cfg(test)]
mod mlx_integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mlx_model_loading() {
        let model = MlxModelBuilder::new("mlx-community/gemma-3-1b-it-4bit")
            .build().await.unwrap();
        
        assert!(model.is_loaded());
    }
    
    #[tokio::test]
    async fn test_basic_generation() {
        let model = load_test_model().await;
        let response = model.generate("Hello").await.unwrap();
        
        assert!(!response.text.is_empty());
        assert!(response.tokens_per_second > 0.0);
    }
}
```

#### B. Chat Template Tests
```rust
#[tokio::test]
async fn test_chat_template_compatibility() {
    let model = MlxModelBuilder::new("mlx-community/gemma-3-1b-it-4bit")
        .build().await.unwrap();
    
    let messages = vec![
        ChatMessage::System("You are a helpful assistant".to_string()),
        ChatMessage::User("What is Rust?".to_string()),
    ];
    
    let response = model.send_chat_request(messages).await.unwrap();
    assert!(!response.text.is_empty());
}
```

#### C. Tool Callback Tests
```rust
#[tokio::test]
async fn test_tool_calling() {
    let calculator = Tool::new("add", |a: f64, b: f64| a + b);
    
    let model = MlxModelBuilder::new("mlx-community/Llama-3.2-3B-8bit")
        .with_tools(vec![calculator])
        .build().await.unwrap();
    
    let response = model.send_chat_request(vec![
        ChatMessage::User("Calculate 15 + 27".to_string())
    ]).await.unwrap();
    
    assert!(response.text.contains("42"));
}
```

#### D. MCP Integration Tests
```rust
#[tokio::test]
async fn test_mcp_compatibility() {
    let model = MlxModelBuilder::new("mlx-community/stable-code-3b-mlx")
        .with_mcp_config(McpConfig::default())
        .build().await.unwrap();
    
    // Test MCP server functionality
    let mcp_server = McpServer::new(model);
    assert!(mcp_server.start().await.is_ok());
}
```

### Performance Benchmarks

```rust
#[tokio::test]
async fn benchmark_mlx_vs_candle() {
    let mlx_model = MlxModelBuilder::new("mlx-community/gemma-3-1b-it-4bit").build().await.unwrap();
    let candle_model = GgufModelBuilder::new("similar-gguf-model").build().await.unwrap();
    
    let prompt = "Write a hello world program in Rust";
    
    let mlx_start = Instant::now();
    let mlx_response = mlx_model.generate(prompt).await.unwrap();
    let mlx_time = mlx_start.elapsed();
    
    let candle_start = Instant::now();
    let candle_response = candle_model.generate(prompt).await.unwrap();
    let candle_time = candle_start.elapsed();
    
    println!("MLX: {:.2?}, Candle: {:.2?}", mlx_time, candle_time);
    assert!(mlx_response.tokens_per_second > 0.0);
}
```

## Implementation Phases

### Phase 1: Core MLX Backend (Week 1-2)
- [ ] Add `mlx-rs` dependency to Cargo.toml
- [ ] Create basic MLX backend structure
- [ ] Implement `MlxModelBuilder` API
- [ ] Basic model loading and inference
- [ ] CLI `mlx` subcommand

### Phase 2: Feature Integration (Week 3-4)
- [ ] Chat template integration
- [ ] Tool callback support
- [ ] ISQ (In-place quantization) support
- [ ] Device management optimization

### Phase 3: Advanced Features (Week 5-6)
- [ ] MCP server/client support
- [ ] Streaming response support
- [ ] Advanced MLX configurations
- [ ] Error handling and logging

### Phase 4: Testing & Documentation (Week 7-8)
- [ ] Comprehensive integration test suite
- [ ] Performance benchmarking
- [ ] Documentation and examples
- [ ] CI/CD integration

## Dependencies

### Required Crates
```toml
[dependencies]
mlx-rs = "0.25"
mlx-macros = "0.25"
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
```

### Platform Requirements
- **Apple Silicon**: M1, M2, M3, M4 Macs
- **macOS**: macOS 12.0+ (Monterey)
- **Rust**: 1.81.0+ (mlx-rs MSRV requirement)
- **Memory**: MLX unified memory architecture

## API Compatibility Matrix

| Feature | GGUF Backend | MLX Backend | Status |
|---------|-------------|-------------|---------|
| Basic Generation | ✅ | 🎯 | Target |
| Chat Templates | ✅ | 🎯 | Target |
| Tool Calling | ✅ | 🎯 | Target |
| MCP Support | ✅ | 🎯 | Target |
| ISQ | ✅ | 🎯 | Target |
| Streaming | ✅ | 🎯 | Target |
| Device Mapping | ✅ | 🎯 | MLX-specific |
| Vision Models | ✅ | 🔄 | Future work |

## Success Criteria

1. **API Parity**: MLX backend provides identical API to existing backends
2. **Performance**: Competitive or better inference speed on Apple Silicon
3. **Compatibility**: All test models load and run correctly
4. **Features**: Chat templates, tools, and MCP work identically
5. **CI/CD**: Automated testing on Apple Silicon runners
6. **Documentation**: Complete examples and integration guides

## Risk Mitigation

### Technical Risks
- **MLX-rs maturity**: Active development project, API may change
  - *Mitigation*: Pin to specific version, contribute upstream
- **Model compatibility**: Not all HF models have MLX versions
  - *Mitigation*: Focus on mlx-community models, provide conversion docs
- **Memory requirements**: Large models still need substantial RAM
  - *Mitigation*: Clear memory requirement documentation

### Integration Risks
- **Backend abstraction**: Adding new backend may require refactoring
  - *Mitigation*: Follow existing patterns, minimal core changes
- **Feature parity**: MLX-specific optimizations may differ from other backends
  - *Mitigation*: Comprehensive test suite ensuring identical behavior

## Future Enhancements

1. **Vision Model Support**: Integrate mlx-vlm for multimodal capabilities
2. **Model Conversion**: Built-in HF → MLX conversion tools
3. **Training Support**: Fine-tuning integration with mlx-rs
4. **Advanced Quantization**: MLX-specific quantization schemes
5. **Mobile Deployment**: iOS/iPadOS support via MLX Swift bindings

---

*This specification provides a comprehensive roadmap for integrating MLX support into mistral.rs while maintaining full compatibility with existing features and ensuring robust testing with appropriate model selection.*
