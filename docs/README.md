# OhMyMiner Documentation Index

Complete technical documentation for the OhMyMiner GPU mining project.

**Last Updated**: November 2, 2025  
**Project Status**: Phase 4B (Golden Vector Validation)  
**Current Focus**: Kernel correctness validation before integration

---

## 📋 Quick Start Documents

Essential reading for understanding current project status:

### **[RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)** (Português) ⭐
Complete executive summary of recent work (Oct 28 - Nov 2, 2025):
- Architecture pivot rationale (cuStateVec → O(1) monolithic kernel)
- Phases 2-4B implementation details
- Quality gates and validation results
- Current blocker and next steps
- **READ THIS FIRST** for project overview

### **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** (English) ⭐
High-level project overview:
- Mission statement and performance targets
- Architecture decision analysis
- Risk assessment and success criteria
- Resource requirements and timeline
- Quality gates status

### **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** �
Comprehensive technical status document:
- All implementation phases documented
- Performance projections by phase
- Technical achievements with code examples
- Memory footprint and computational complexity
- Critical path forward

### **[PHASE_4B_GOLDEN_VECTORS.md](PHASE_4B_GOLDEN_VECTORS.md)** 🔬
Golden vector validation guide:
- Test input requirements
- Expected output specifications
- Extraction strategies (QTC reference, CPU simulator)
- Validation workflow and success criteria
- **CRITICAL** for unblocking current phase

### **[RECENT_WORK.md](RECENT_WORK.md)** 📝
Detailed changelog of recent implementation:
- Phase-by-phase breakdown (Oct 28 - Nov 2)
- Files modified/created with line counts
- Build fixes and test results
- Quality metrics summary

---

## 📖 Technical Documentation

### Algorithm & Architecture

#### **[ANALYSIS_REFERENCE_QHASH.md](ANALYSIS_REFERENCE_QHASH.md)**
Deep dive into qhash proof-of-work algorithm:
- Quantum circuit structure (16 qubits, 2 layers, 72 gates)
- Parametrization from hash (nibble extraction)
- Fixed-point Q15 consensus format
- Complete pipeline analysis

#### **[batching-analysis.md](batching-analysis.md)**
Batching strategy and memory optimization:
- O(1) VRAM architecture analysis
- Memory layout per block
- Batch size calculations
- Pipeline optimization strategies

#### **[cucomplex-types.md](cucomplex-types.md)**
CUDA complex number handling:
- cuDoubleComplex operations
- Amplitude representation
- Gate matrix operations

### Build & Setup

#### **[INSTALL_CUQUANTUM.md](INSTALL_CUQUANTUM.md)**
cuQuantum SDK installation guide (legacy):
- Installation steps
- Environment configuration
- Build system integration
- **Note**: cuStateVec approach superseded, but SDK may be useful for validation

### Reference Material

#### **[qtc-doc.md](qtc-doc.md)**
Qubitcoin documentation reference:
- Protocol specifications
- Block structure
- Mining parameters

---

## 📦 Archived Documents

**Location**: `archive/`

Documents containing superseded approaches (cuStateVec integration attempts):

- **[archive/cuquantum-integration.md](archive/cuquantum-integration.md)** - Initial cuStateVec backend
- **[archive/cuquantum-optimization-summary.md](archive/cuquantum-optimization-summary.md)** - Optimization attempts
- **[archive/cuquantum-batching-optimization.md](archive/cuquantum-batching-optimization.md)** - Batched cuStateVec
- **[archive/critical-discovery-cuquantum.md](archive/critical-discovery-cuquantum.md)** - O(2^n) bottleneck discovery
- **[archive/README.md](archive/README.md)** - Archive index with historical context

**Why Archived**: O(2^n) VRAM scaling made approach impractical. Replaced by O(1) monolithic kernel.

---

## 🎯 Quick Navigation

| Task | Document |
|------|----------|
| **Project overview** | [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md) (PT) or [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) (EN) |
| **Current status** | [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) |
| **Unblock Phase 4B** | [PHASE_4B_GOLDEN_VECTORS.md](PHASE_4B_GOLDEN_VECTORS.md) |
| **Recent work** | [RECENT_WORK.md](RECENT_WORK.md) |
| **Algorithm details** | [ANALYSIS_REFERENCE_QHASH.md](ANALYSIS_REFERENCE_QHASH.md) |
| **Batching strategy** | [batching-analysis.md](batching-analysis.md) |
| **Historical context** | [archive/README.md](archive/README.md) |

---

## 📊 Documentation Status

| Document | Status | Last Updated | Purpose |
|----------|--------|--------------|---------|
| RESUMO_EXECUTIVO.md | ✅ Current | Nov 2, 2025 | Executive summary (PT) |
| EXECUTIVE_SUMMARY.md | ✅ Current | Nov 2, 2025 | Executive summary (EN) |
| IMPLEMENTATION_STATUS.md | ✅ Current | Nov 2, 2025 | Technical status |
| PHASE_4B_GOLDEN_VECTORS.md | ✅ Current | Nov 2, 2025 | Validation guide |
| RECENT_WORK.md | ✅ Current | Nov 2, 2025 | Recent changelog |
| ANALYSIS_REFERENCE_QHASH.md | ✅ Current | - | Algorithm reference |
| batching-analysis.md | ✅ Current | - | Batching strategy |
| archive/* | 📚 Historical | Oct 2025 | Superseded approaches |

---

## 🔄 Maintenance

This documentation is actively maintained alongside code implementation.

**Current Phase**: Phase 4B (Golden Vector Validation)  
**Next Update**: After Phase 4B completion with validation results

---

**Note**: This is a GPU-only miner. All CPU mining references in archived documents are obsolete.
