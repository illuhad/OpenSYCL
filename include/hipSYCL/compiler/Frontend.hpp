/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_FRONTEND_HPP
#define HIPSYCL_FRONTEND_HPP

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <regex>
#include <sstream>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Sema/Sema.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "CompilationState.hpp"
#include "Attributes.hpp"

#include "hipSYCL/common/debug.hpp"


namespace hipsycl {
namespace compiler {
namespace detail {

///
/// Utility type to generate the set of all function declarations
/// implicitly or explicitly reachable from some initial declaration.
///
/// NOTE: Must only be used when the full translation unit is present,
/// e.g. in HandleTranslationUnitDecl, otherwise the callset
/// might not be complete.
///
class CompleteCallSet : public clang::RecursiveASTVisitor<CompleteCallSet> {
  public:
    using FunctionSet = std::unordered_set<clang::FunctionDecl*>;

    explicit CompleteCallSet(clang::Decl* D)
    {
      TraverseDecl(D);
    }

    bool VisitFunctionDecl(clang::FunctionDecl* FD)
    {
      visitedDecls.insert(FD);
      return true;
    }

    bool VisitCallExpr(clang::CallExpr* CE)
    {
      if(auto Callee = CE->getDirectCallee())
        TraverseDecl(Callee);
      return true;
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr* CE)
    {
      if(auto Callee = CE->getConstructor())
      {
        TraverseDecl(Callee);
        // Since for destructor calls no explicit AST nodes are created, we simply use this opportunity to
        // find the corresponding destructor for all constructed types (since we assume that every
        // type that can be constructed on the GPU also can and will be destructed).
        if(auto Ptr = llvm::dyn_cast_or_null<clang::PointerType>(Callee->getThisType()->getCanonicalTypeUnqualified()))
          if(auto Record = llvm::dyn_cast<clang::RecordType>(Ptr->getPointeeType()))
            if(auto RecordDecl = llvm::dyn_cast<clang::CXXRecordDecl>(Record->getDecl()))
              if(auto DtorDecl = RecordDecl->getDestructor())
                TraverseDecl(DtorDecl);
      }
      return true;
    }

    bool TraverseDecl(clang::Decl* D)
    {
      // fixme: investigate where the invalid decls come from..
      if(!D)
        return true;
      
      clang::Decl* DefinitionDecl = D;
      clang::FunctionDecl* FD = clang::dyn_cast<clang::FunctionDecl>(D);

      if(FD){
        const clang::FunctionDecl* ActualDefinition;
        if(FD->isDefined(ActualDefinition)) {
          
          DefinitionDecl = const_cast<clang::FunctionDecl*>(ActualDefinition);
        }
      }

      if (visitedDecls.find(llvm::dyn_cast_or_null<clang::FunctionDecl>(
              DefinitionDecl)) == visitedDecls.end())
        return clang::RecursiveASTVisitor<CompleteCallSet>::TraverseDecl(
            DefinitionDecl);
      
      return true;
    }

    bool shouldWalkTypesOfTypeLocs() const { return false; }
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool shouldVisitImplicitCode() const { return true; }

    const FunctionSet& getReachableDecls() const { return visitedDecls; }

  private:
    FunctionSet visitedDecls;
};

///
/// Builds a kernel name from a RecordDecl, taking into account template specializations.
/// Returns an empty string if the name is not a valid kernel name.
///
inline std::string buildKernelNameFromRecordType(const clang::QualType &RecordType, clang::MangleContext *Mangler) {
  std::string KernelName;
  llvm::raw_string_ostream SS(KernelName);
#if LLVM_VERSION_MAJOR >= 18
  Mangler->mangleCanonicalTypeName(RecordType, SS);
#else
  Mangler->mangleTypeName(RecordType, SS);
#endif

  return KernelName;
}

inline std::string buildKernelName(clang::RecordDecl* D, clang::MangleContext *Mangler) {
  assert(D);
  assert(Mangler);
  auto DeclName = buildKernelNameFromRecordType(
      Mangler->getASTContext().getTypeDeclType(D), Mangler);
  return "__acpp_kernel_" + DeclName;
}

// Partially taken from CGCUDANV.cpp
inline std::string
getDeviceSideName(clang::NamedDecl *ND, clang::ASTContext &Ctx,
                  clang::MangleContext *RegularMangleContext,
                  clang::MangleContext *DeviceMangleContext) {
  clang::GlobalDecl GD;
  // D could be either a kernel or a variable.
  if (auto *FD = clang::dyn_cast<clang::FunctionDecl>(ND))
    GD = clang::GlobalDecl(FD, clang::KernelReferenceKind::Kernel);
  else
    GD = clang::GlobalDecl(ND);

  std::string DeviceSideName;
  static clang::MangleContext *MC = nullptr;
  if(!MC) {
    if (Ctx.getLangOpts().CUDAIsDevice){
      assert(RegularMangleContext);
      MC = RegularMangleContext;
    }
    else {
      assert(DeviceMangleContext);
      MC = DeviceMangleContext;
    }
  }
  if (MC->shouldMangleDeclName(ND)) {
    llvm::SmallString<256> Buffer;
    llvm::raw_svector_ostream Out(Buffer);
    MC->mangleName(GD, Out);
    DeviceSideName = std::string(Out.str());
  } else
    DeviceSideName = std::string(ND->getIdentifier()->getName());

  return DeviceSideName;
}
}

class FrontendASTVisitor : public clang::RecursiveASTVisitor<FrontendASTVisitor>
{
  clang::CompilerInstance &Instance;
  

public:
  FrontendASTVisitor(clang::CompilerInstance &instance)
      : Instance{instance} {

    clang::MangleContext* NameMangler = nullptr;
    clang::MangleContext* DeviceNameMangler = nullptr;

    // On clang 13+, we rely on kernel name mangling powered by CUDA/HIP
    // support in clang and __builtin_get_device_side_mangled_name() in client code.
    // For this, we need to have a regular mangling context in the device pass,
    // and an explicit device mangler in the host pass.
    if(instance.getLangOpts().CUDAIsDevice)
      NameMangler = instance.getASTContext().createMangleContext();
    else
      // For legacy mangling (e.g. -D__ACPP_SPLIT_COMPILER__) force Itanium ABI
      // also in the host pass. Non-legacy mangling would use the DeviceNameMangler
      // anyway in the host pass.
      NameMangler = clang::ItaniumMangleContext::create(
        instance.getASTContext(), instance.getASTContext().getDiagnostics());

    // DeviceNameMangler is only used during the host pass
    if (instance.getAuxTarget() && instance.getTarget().getCXXABI().isMicrosoft() &&
        instance.getAuxTarget()->getCXXABI().isItaniumFamily()) {
      DeviceNameMangler =
          instance.getASTContext().createDeviceMangleContext(*instance.getAuxTarget());
    } else {
      DeviceNameMangler =
          instance.getASTContext().createMangleContext(instance.getASTContext().getAuxTargetInfo());
    }

    KernelNameMangler.reset(NameMangler);
    DeviceKernelNameMangler.reset(DeviceNameMangler);
  }

  ~FrontendASTVisitor()
  {}
  
  bool shouldVisitTemplateInstantiations() const { return true; }

  /// Return whether this visitor should recurse into implicit
  /// code, e.g., implicit constructors and destructors.
  bool shouldVisitImplicitCode() const { return true; }
  
  // We also need to have look at all statements to identify Lambda declarations
  bool VisitStmt(clang::Stmt *S) {

    if(clang::isa<clang::LambdaExpr>(S))
    {
      clang::LambdaExpr* lambda = clang::cast<clang::LambdaExpr>(S);
      clang::FunctionDecl* callOp = lambda->getCallOperator();
      if(callOp)
        this->VisitFunctionDecl(callOp);
    }
    
    return true;
  }
  
  bool VisitDecl(clang::Decl* D){
    if(clang::VarDecl* V = clang::dyn_cast<clang::VarDecl>(D)){
      if(isLocalMemory(V))
        // Maybe we should additionally check that this is in kernels?
        storeVariableInLocalMemory(V);
    }
    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *f) {
    if(!f)
      return true;

    this->processFunctionDecl(f);

    return true;
  }

  bool VisitCallExpr(clang::CallExpr *Call) {
    auto F = llvm::dyn_cast_or_null<clang::FunctionDecl>(Call->getDirectCallee());
    if(!F)
      return true;

    if(F->getQualifiedNameAsString() == "__acpp_kernel_name_template") {
      return handleKernelStub(F);
    } else if (CustomAttributes::SyclKernel.isAttachedTo(F)){

      auto KernelFunctorType = llvm::dyn_cast<clang::RecordType>(Call->getArg(0)
        ->getType()->getCanonicalTypeUnqualified());

      return handleKernel(F, KernelFunctorType);
    }

    return true;
  }

  void applyAttributes()
  {
    for(auto F : MarkedHostDeviceFunctions)
    {
      // Strictly speaking, setting these attributes is not even necessary!
      // It's only important that the kernel has the right attribute.
      if (!F->hasAttr<clang::CUDAHostAttr>() && !F->hasAttr<clang::CUDADeviceAttr>())
      {
        F->addAttr(clang::CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
        F->addAttr(clang::CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));
      }
    }

    for(auto F : MarkedKernels)
    {
      if (!F->hasAttr<clang::CUDAGlobalAttr>() &&
          CustomAttributes::SyclKernel.isAttachedTo(F)) {

        auto* NewAttr = clang::CUDAGlobalAttr::CreateImplicit(Instance.getASTContext());

        F->addAttr(NewAttr);

        // create amdgpu_flat_work_group_size attribute to allow sub_groups outside of [128, 256]
        // first we need to create Expressions containing the workgroup-sizes
        auto sizeType = Instance.getASTContext().getSizeType();
        auto minFlatWorkgroupSize = llvm::APInt(Instance.getASTContext().getTypeSize(sizeType), 64);
        auto maxFlatWorkgroupSize = llvm::APInt(Instance.getASTContext().getTypeSize(sizeType), 1024);
        auto* minFlatWorkgroupExpr = clang::IntegerLiteral::Create(Instance.getASTContext(), minFlatWorkgroupSize, sizeType, clang::SourceLocation{});
        auto* maxFlatWorkgroupExpr = clang::IntegerLiteral::Create(Instance.getASTContext(), maxFlatWorkgroupSize, sizeType, clang::SourceLocation{});

        // to finally create the attribute itself
        auto * NewAMDGPUAttr = clang::AMDGPUFlatWorkGroupSizeAttr::CreateImplicit(Instance.getASTContext(), minFlatWorkgroupExpr , maxFlatWorkgroupExpr);

        F->addAttr(NewAMDGPUAttr);
      }
    }

    for(auto F : UserKernels)
    {
      // Mark all functions called by user kernels as host / device.
      detail::CompleteCallSet CCS(F);
      for (auto&& RD : CCS.getReachableDecls())
      {
        HIPSYCL_DEBUG_INFO << "AST processing: Marking function as __host__ __device__: "
                           << RD->getQualifiedNameAsString() << "\n";
        markAsHostDevice(RD);
        if (!RD->hasAttr<clang::CUDAHostAttr>() && !RD->hasAttr<clang::CUDADeviceAttr>()
            && !CustomAttributes::SyclKernel.isAttachedTo(RD))
        {
          RD->addAttr(clang::CUDAHostAttr::CreateImplicit(Instance.getASTContext()));
          RD->addAttr(clang::CUDADeviceAttr::CreateImplicit(Instance.getASTContext()));
        }
      }

      // Rename kernel according to kernel name tag and body
      nameKernel(F);
    }

    for(auto* Kernel : HierarchicalKernels){
      HIPSYCL_DEBUG_INFO << "AST Processing: Detected parallel_for_workgroup kernel "
                        << Kernel->getQualifiedNameAsString() << "\n";

      // Mark local variables as shared memory, unless they are explicitly marked private.
      // Do this not only for the kernel itself, but consider all functions called by the kernel.
      detail::CompleteCallSet CCS(Kernel);
      for(auto&& RD : CCS.getReachableDecls())
      {
        // To prevent every local variable in any function being marked as shared,
        // we only consider functions that receive a hipsycl::sycl::group as their parameter.
        for(auto Param = RD->param_begin(); Param != RD->param_end(); ++Param)
        {
          auto Type = (*Param)->getOriginalType().getTypePtr();
          if(auto DeclType = Type->getAsCXXRecordDecl()) {
            if(DeclType->getQualifiedNameAsString() == "hipsycl::sycl::group")
            {
              storeLocalVariablesInLocalMemory(RD->getBody(), RD);
              break;
            }
          }
        }
      }
    }

    auto MakeKernelsNoexcept = [&](clang::FunctionDecl* F) {
      detail::CompleteCallSet CCS(F);
      for (auto &D : CCS.getReachableDecls()) {
        if (!clang::isNoexceptExceptionSpec(D->getExceptionSpecType())) {
          HIPSYCL_DEBUG_INFO << "AST processing: Marking function as noexcept: " << D->getQualifiedNameAsString()
                             << "\n";
          D->addAttr(clang::NoThrowAttr::CreateImplicit(Instance.getASTContext()));
        }
      }
    };

    for(auto* F : HostNDKernels) {
      MakeKernelsNoexcept(F);
    }
    for(auto* F : SSCPOutliningEntrypoints) {
      MakeKernelsNoexcept(F);
    }
  }

  std::unordered_set<clang::FunctionDecl*>& getMarkedHostDeviceFunctions()
  {
    return MarkedHostDeviceFunctions;
  }

  std::unordered_set<clang::FunctionDecl*>& getKernels()
  {
    return MarkedKernels;
  }

private:
  std::unordered_set<clang::FunctionDecl*> MarkedHostDeviceFunctions;
  std::unordered_set<clang::FunctionDecl*> MarkedKernels;
  std::unordered_set<clang::FunctionDecl*> HierarchicalKernels;

  std::unordered_set<clang::FunctionDecl*> UserKernels;
  // Maps a Kernel name tag or kernel body type to the mangled name
  // of a kernel stub function
  std::unordered_map<const clang::RecordType*, clang::FunctionDecl*> KernelManglingNameTemplates;
  // Maps the declaration/instantiation of a kernel to the kernel body
  // (kernel lambda or function object)
  std::unordered_map<clang::FunctionDecl*, const clang::RecordType*> KernelBodies;

  std::unordered_set<clang::FunctionDecl*> HostNDKernels;
  std::unordered_set<clang::FunctionDecl*> SSCPOutliningEntrypoints;

  std::unique_ptr<clang::MangleContext> KernelNameMangler;
  // Only used on clang 13+. Name mangler that takes into account
  // the device numbering of kernel lambdas.
  std::unique_ptr<clang::MangleContext> DeviceKernelNameMangler;

  void markAsHostDevice(clang::FunctionDecl* F)
  {
    this->MarkedHostDeviceFunctions.insert(F);
  }

  void markAsKernel(clang::FunctionDecl* F)
  {
    this->MarkedKernels.insert(F);
  }

  void markAsNDKernel(clang::FunctionDecl* F)
  {
    this->HostNDKernels.insert(F);
  }

  void markAsSSCPOutliningEntrypoint(clang::FunctionDecl* F)
  {
    this->SSCPOutliningEntrypoints.insert(F);
  }

  void processFunctionDecl(clang::FunctionDecl* f)
  {
    if(!f)
      return;

    if(f->getQualifiedNameAsString()
        == "hipsycl::glue::hiplike_dispatch::parallel_for_workgroup")
    {
      clang::FunctionDecl* Kernel = f;
      
      HierarchicalKernels.insert(Kernel);
    }
  
    
    if(CustomAttributes::SyclKernel.isAttachedTo(f)){
      markAsKernel(f); 
    }

    // Need to iterate over all attributes to support the case
    // where multiple annotate attributes are present.
    if(f->hasAttrs()) {
      for(auto* Attr : f->getAttrs()) {
        if(auto* AAttr = clang::dyn_cast<clang::AnnotateAttr>(Attr)) {
          if (AAttr->getAnnotation() == "hipsycl_nd_kernel") {
            markAsNDKernel(f);
          } else if (AAttr->getAnnotation() == "hipsycl_sscp_outlining") {
            markAsSSCPOutliningEntrypoint(f);
          }
        }
      }
    }
  }

  bool isPrivateMemory(const clang::VarDecl* V) const
  {
    const clang::CXXRecordDecl* R = V->getType()->getAsCXXRecordDecl();
    if(R)
      return R->getQualifiedNameAsString() == "hipsycl::sycl::private_memory";
  
    return false;
  }

  bool isLocalMemory(const clang::VarDecl* V) const
  {
    const clang::CXXRecordDecl* R = V->getType()->getAsCXXRecordDecl();
    if(R)
      return R->getQualifiedNameAsString() == "hipsycl::sycl::local_memory";
  
    return false;
  }

  ///
  /// Marks all variable declarations within a given block statement as shared memory,
  /// unless they are explicitly declared as a private memory type.
  ///
  /// Recurses into compound statements (i.e., a set of braces {}).
  ///
  /// NOTE TODO: It is unclear how certain other statement types should be handled.
  /// For example, should the loop variable of a for-loop be marked as shared? Probably not.
  ///
  void storeLocalVariablesInLocalMemory(clang::Stmt* BlockStmt, clang::FunctionDecl* F) const
  {
    for(auto S = BlockStmt->child_begin(); S != BlockStmt->child_end(); ++S)
    {
      if(auto D = clang::dyn_cast<clang::DeclStmt>(*S))
      {
        for(auto decl = D->decl_begin(); decl != D->decl_end(); ++decl)
        {
          if(clang::VarDecl* V = clang::dyn_cast<clang::VarDecl>(*decl))
          {
            if(!isPrivateMemory(V))
              storeVariableInLocalMemory(V);
          }
        }
      }
      else if(clang::dyn_cast<clang::CompoundStmt>(*S))
      {
        storeLocalVariablesInLocalMemory(*S, F);
      }
    }
  }
  
  void storeVariableInLocalMemory(clang::VarDecl* V) const {
    HIPSYCL_DEBUG_INFO
                  << "AST Processing: Marking variable "
                  << V->getNameAsString()
                  << " as __shared__"
                  << "\n";

    if (!V->hasAttr<clang::CUDASharedAttr>()) {
      V->addAttr(clang::CUDASharedAttr::CreateImplicit(
          Instance.getASTContext()));
      V->setStorageClass(clang::SC_Static);
    }
  }


  const clang::RecordType* getTemplateTypeArgument(clang::FunctionDecl* F, int TemplateArg) {
    clang::FunctionTemplateSpecializationInfo* Info = F->getTemplateSpecializationInfo();

    if(Info) {
      if(TemplateArg >= Info->TemplateArguments->size())
        return nullptr;

      const auto KernelNameArgument = Info->TemplateArguments->get(TemplateArg);

      if (KernelNameArgument.getKind() == clang::TemplateArgument::ArgKind::Type) {
        if (auto RecordType = llvm::dyn_cast<clang::RecordType>(
                KernelNameArgument.getAsType().getTypePtr())) {
          return RecordType;
        }
      }
    }
    return nullptr;
  }

  const clang::RecordType* getKernelNameTag(clang::FunctionDecl* F) {
    return getTemplateTypeArgument(F, 0);
  }

  bool isKernelUnnamed(clang::FunctionDecl* F) {
    if(!F)
      return false;

    const clang::RecordType* NameTag = getKernelNameTag(F);

    if(!NameTag)
      // If name tag is invalid, assume unnamed
      return true;

    if(NameTag->getDecl()) {
      return NameTag->getDecl()->getQualifiedNameAsString() ==
             "__acpp_unnamed_kernel";
    }

    return true;
  }

  // Returns either kernel name tag or kernel body, depending on whether
  // the kernel is named or unnamed
  const clang::RecordType* getRelevantKernelNamingComponent(clang::FunctionDecl* F) {
    if(isKernelUnnamed(F)) {
      auto BodyIterator = KernelBodies.find(F);

      if(BodyIterator == KernelBodies.end()) {
        HIPSYCL_DEBUG_ERROR
            << "Kernel did not have body registered, this should never happen\n";
        return nullptr;
      }

      return BodyIterator->second;
    } else {
      return getKernelNameTag(F);
    }
  }

  // Should be invoked whenever a call to __acpp_hiplike_kernel stub is encountered.
  // These functions are only used to borrow demangleable kernel names in the form
  // __acpp_hiplike_kernel<KernelName>
  //
  // The kernel stubs are only used to generate mangled names
  // that can then be copied to the actual kernels.
  //
  // This is mainly used on clang 13+ where __builtin_get_device_side_mangled_name()
  // is available, but requires an actual __global__ function on which to operate.
  bool handleKernelStub(clang::FunctionDecl* F) {

    if(!isKernelUnnamed(F)) {
      const clang::RecordType* KernelNameTag = getKernelNameTag(F);

      if(KernelNameTag) {
        KernelManglingNameTemplates[KernelNameTag] = F;
      }
    }

    return true;
  }

  bool handleKernel(clang::FunctionDecl* F, const clang::RecordType* KernelBody) {
    UserKernels.insert(F);
    KernelBodies[F] = KernelBody;
    return true;
  }

  void setKernelName(clang::FunctionDecl* F, const std::string& name) {

    // Abort with error diagnostic if no kernel name could be built
    if(name.empty())
    {
      // Try to get the declaration of the kernel functor for error
      // diagnostics
      auto B = KernelBodies.find(F);
      clang::Decl* ErrorDecl = F;

      if(B != KernelBodies.end()) {
        ErrorDecl = B->second->getDecl();
      }

      auto SL = ErrorDecl->getSourceRange().getBegin();
      auto ID = Instance.getASTContext().getDiagnostics()
        .getCustomDiagID(clang::DiagnosticsEngine::Level::Error,
            "No valid kernel name for kernel submission");
      Instance.getASTContext().getDiagnostics().Report(SL, ID);
    }

    // Add the AsmLabel attribute which, if present,
    // is used by Clang instead of the function's mangled name.
    if(!F->hasAttr<clang::AsmLabelAttr>()) {
      F->addAttr(
          clang::AsmLabelAttr::CreateImplicit(Instance.getASTContext(), name));
      HIPSYCL_DEBUG_INFO << "AST processing: Adding ASM label attribute with kernel name "
        << name << "\n";
    }
  }

  void nameKernelUsingTypes(clang::FunctionDecl* F, bool RenameUnnamedKernels) {
    std::string KernelName;

    // If we are dealing with a named kernel, construct the name
    // based on the kernel name argument
    if(!isKernelUnnamed(F)) {
      KernelName = detail::buildKernelName(getKernelNameTag(F)->getAsRecordDecl(),
                                            KernelNameMangler.get());
      setKernelName(F, KernelName);
    } else {
      // In certain configurations, we can just let clang handle the naming
      // of unnamed kernels
      if(RenameUnnamedKernels) {
        // Otherwise (for unnamed kernels or non-lambda kernels)
        // construct name based on kernel functor type.

        const auto KernelFunctorArgument = getTemplateTypeArgument(F, 1);

        if (KernelFunctorArgument) {
          KernelName = detail::buildKernelName(KernelFunctorArgument->getAsRecordDecl(),
                                                KernelNameMangler.get());
          setKernelName(F, KernelName);
        }
      }
    }
  }

  // LLVM 11 supports __builtin_unique_stable_name() and unique
  // name manglers. Rely on those to support all mangling
  void nameKernelUsingUniqueMangler(clang::FunctionDecl* F) {
    // We just need to enforce that all names (even kernels without explicit names)
    // are mangled with our unique mangler, so pass "true" to force renaming all
    // kernels.
    nameKernelUsingTypes(F, true);
  }

  void nameKernelUsingKernelManglingStub(clang::FunctionDecl* F) {
    const clang::RecordType* NamingComponent = getRelevantKernelNamingComponent(F);
    auto SuggestionIt = KernelManglingNameTemplates.find(NamingComponent);

    if(SuggestionIt == KernelManglingNameTemplates.end()) {
      HIPSYCL_DEBUG_ERROR << "Did not find kernel mangling suggestion for "
                             "encountered kernel, this should never happen.\n";
    }

    std::string KernelName = detail::getDeviceSideName(
        SuggestionIt->second, Instance.getASTContext(), KernelNameMangler.get(),
        DeviceKernelNameMangler.get());

    std::string TemplateMarker = "_Z27__acpp_kernel_name_template";
    std::string Replacement = "_Z13__acpp_kernel";
    assert(KernelName.size() > TemplateMarker.size());
    KernelName.erase(0, TemplateMarker.size());
    KernelName = Replacement + KernelName;

    setKernelName(F, KernelName);
  }

  void nameKernel(clang::FunctionDecl* F) {

    auto KernelFunctorType = KernelBodies[F];

    // Starting with clang 13, we rely on mangling by borrowing
    // the name from a mangling stub in the glue code, which
    // has the advantage that it
    // a) is demangleable
    // b) can be queried from client code using __builtin_get_device_side_mangled_name()
    //
    // However, this only makes sense if the client code has access to this builtin.
    // In a split compilation scenario, where clang is not the host compiler, this will
    // not be the case. In this situation we need to again mangle using the types of name tag
    // and kernel body so that client code will at least be able to use typeid().
    // In such a split compilation scenario, unnamed kernel lambdas are unsupported.
    bool IsSplitCompilerConfiguration = false;
    for(auto V : Instance.getPreprocessor().getPreprocessorOpts().Macros) {
      if(V.first == "__ACPP_SPLIT_COMPILER__")
        IsSplitCompilerConfiguration = true;
    }
    if(IsSplitCompilerConfiguration) {
      // Unnamed kernel lambdas are unsupported in split compiler configuration, emit
      // error if this happens
      if(isKernelUnnamed(F)) {
        if(clang::CXXRecordDecl* KernelBody = KernelFunctorType->getAsCXXRecordDecl()) {

          if(KernelBody->isLambda()) {
            if (KernelFunctorType->getAsCXXRecordDecl() &&
                KernelFunctorType->getAsCXXRecordDecl()->isLambda()) {
              auto SL = llvm::dyn_cast<clang::CXXRecordDecl>(
                            KernelFunctorType->getDecl())
                            ->getSourceRange()
                            .getBegin();
              auto ID =
                  Instance.getASTContext().getDiagnostics().getCustomDiagID(
                      clang::DiagnosticsEngine::Level::Error,
                      "Unnamed kernel lambdas are unsupported in a split "
                      "compilation configuration where the host compiler is "
                      "not clang.");
              Instance.getASTContext().getDiagnostics().Report(SL, ID);
            }
          }

        }
      }
      nameKernelUsingTypes(F, true);
    }
    else {
      nameKernelUsingKernelManglingStub(F);
    }
  }

};




class FrontendASTConsumer : public clang::ASTConsumer {
  
  FrontendASTVisitor Visitor;
  clang::CompilerInstance& Instance;
  
public:
  FrontendASTConsumer(clang::CompilerInstance &I)
      : Visitor{I}, Instance{I}
  {
    CompilationStateManager::get().reset();
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override {
    for (auto&& D : DG)
      Visitor.TraverseDecl(D);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    
    CompilationStateManager::getASTPassState().setDeviceCompilation(
        Instance.getSema().getLangOpts().CUDAIsDevice);

    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __device__ ****** " << "\n";
    else
      HIPSYCL_DEBUG_INFO << " ****** Entering compilation mode for __host__ ****** " << "\n";

    Visitor.applyAttributes();

    // The following part is absolutely crucial:
    //
    // clang works roughly like this when building and processing the AST:
    // while(!not done){
    //   DeclGroupRef DG = parseNextDeclGroup()
    //   foreach(ASTConsumer C){
    //     C->HandleTopLevelDecl(DG);
    //   }
    // }
    // }
    // foreach(ASTConsumer C){
    //   C->HandleTranslationUnit(TU);
    // }
    //
    // The BackendConsumers which take care of emitting IR code
    // already emit in HandleTopLevelDecl().
    // This means that, since we have only made attribute changes in
    // HandleTranslationUnit(), all code has already been emitted without taking
    // into account our changes. In particular, since functions used in SYCL
    // kernels hadn't yet been marked as __device__ at this point, none of them
    // actually got emitted. To fix this, we let all registered ASTConsumers run
    // their HandleTopLevelDecl() over the functions where we have added
    // attributes. Since we do not implement HandleTopLevelDecl(), the only
    // consumers affected are the BackendConsumers which will then generate the
    // required IR for device code.
    clang::ASTConsumer& C = Instance.getASTConsumer();
    if(clang::isa<clang::MultiplexConsumer>(&C))
    {
      clang::MultiplexConsumer& MC = static_cast<clang::MultiplexConsumer&>(C);
      if(CompilationStateManager::getASTPassState().isDeviceCompilation()){

        for (clang::FunctionDecl *HDFunction :
            Visitor.getMarkedHostDeviceFunctions()) {
          clang::DeclGroupRef DG{HDFunction};

          MC.HandleTopLevelDecl(DG);
        }
      }
      // We need to reemit kernels both in host and device passes
      // to make sure the right stubs are generated
      for(clang::FunctionDecl* Kernel : Visitor.getKernels()){
          clang::DeclGroupRef DG{Kernel};

          MC.HandleTopLevelDecl(DG);
      }
    }
  }
};

}
}

#endif
