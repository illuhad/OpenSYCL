#include "hipSYCL/compiler/GlobalsPruningPass.hpp"

namespace {
bool canGlobalVariableBeRemoved(llvm::GlobalVariable *G) {
  G->removeDeadConstantUsers();
  return G->getNumUses() == 0;
}

void pruneUnusedGlobals(llvm::Module &M) {

  HIPSYCL_DEBUG_INFO << " ****** Starting pruning of global variables ******\n";

  std::vector<llvm::GlobalVariable *> VariablesForPruning;

  for (auto G = M.global_begin(); G != M.global_end(); ++G) {
    llvm::GlobalVariable *GPtr = &(*G);
    if (canGlobalVariableBeRemoved(GPtr)) {
      VariablesForPruning.push_back(GPtr);

      HIPSYCL_DEBUG_INFO
          << "IR Processing: Pruning unused global variable from device code: "
          << G->getName().str() << "\n";
    }
  }

  for (auto G : VariablesForPruning) {
    G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
    G->eraseFromParent();
  }
  HIPSYCL_DEBUG_INFO << "===> IR Processing: Pruning of globals complete, removed "
                     << VariablesForPruning.size() << " global variable(s).\n";
}

} // namespace

bool hipsycl::compiler::GlobalsPruningPassLegacy::runOnModule(llvm::Module &M) {
  if (!CompilationStateManager::getASTPassState().isDeviceCompilation())
    return false;

  pruneUnusedGlobals(M);

  return true;
}

#if !defined(_WIN32) || defined(ACPP_LLVM_COMPONENT)
llvm::PreservedAnalyses
hipsycl::compiler::GlobalsPruningPass::run(llvm::Module &M,
                                           llvm::ModuleAnalysisManager &AM) {
  if (!CompilationStateManager::getASTPassState().isDeviceCompilation())
    return llvm::PreservedAnalyses::all();

  pruneUnusedGlobals(M);

  // todo: be a bit more granular
  return llvm::PreservedAnalyses::none();
}
#endif // !_WIN32

char hipsycl::compiler::GlobalsPruningPassLegacy::ID = 0;
