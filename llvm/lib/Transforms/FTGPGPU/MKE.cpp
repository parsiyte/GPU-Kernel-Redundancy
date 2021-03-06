
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>

#include "FTGPGPUPass.cpp"

using namespace llvm;


class MKE : public AbstractPass{
  
  public:
  
    MKE(bool StreamEnabled = false) { this->StreamEnabled = StreamEnabled;}
    bool StreamEnabled;
    
    bool executeThePass(CallInst* FunctionCall, Auxiliary* PassAuxiliary) const override{
      BasicBlock* CurrentBB = FunctionCall->getParent();
      BasicBlock* NextBB = CurrentBB->getNextNode();
      BasicBlock* PrevBB = CurrentBB->getPrevNode();
      Value* StreamArray;
      Function* FunctionToReplicate = FunctionCall->getCalledFunction();
      FunctionCallee StreamCreateFunction = PassAuxiliary->StreamCreateFunction;

      Output* OutputObject = PassAuxiliary->OutputObject;
      bool IsLoop = isForLoop(PrevBB);

      Instruction* FirstInstructionOfNextBB = NextBB->getFirstNonPHI();
      Instruction* FirstInstructionOfPrevBB = PrevBB->getFirstNonPHI();
      Instruction& LastInstructionOfPrevBB = PrevBB->back();
      IRBuilder<> Builder(FunctionCall); 
      Value* Zero32bit = PassAuxiliary->Zero32Bit;
      Value* One32bit = PassAuxiliary->One32Bit;
      Type* Int32Type = PassAuxiliary->Int32Type;
      Value* CudaStreamNonBlocking = PassAuxiliary->CudaStreamNonBlocking;
      if(this->StreamEnabled == true){
        ArrayType *ArrayType = ArrayType::get(PassAuxiliary->StreamType, NumberOfReplication);
        Builder.SetInsertPoint(FirstInstructionOfPrevBB);
        StreamArray = Builder.CreateAlloca(ArrayType, nullptr, "streams");
        
        Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, Zero32bit}, "arrayidx"); // Bunu zaten özgün çağrıya verdiğimiz için direkt 0 verebiliriz.
        Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
        Value* LoadedStream = Builder.CreateLoad(IthStream);
        PassAuxiliary->CudaConfigureCall->setArgOperand(StreamArgIndex, LoadedStream);
      }
      Builder.SetInsertPoint(&LastInstructionOfPrevBB);

      std::pair<Value*, Value* > Errors = createErrorValue(Builder, LastInstructionOfPrevBB, PassAuxiliary);
      Instruction* CheckPoint = createCheckPoint(Builder, OutputObject, LastInstructionOfPrevBB, PassAuxiliary);
      createAndAllocateVariableAndreMemCpy(Builder, OutputObject, LastInstructionOfPrevBB, PassAuxiliary->CudaMemCopy, IsLoop, 0);


        CallInst* ClonedConfigureCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfigureCall->clone());
        ClonedConfigureCall->insertBefore(FirstInstructionOfNextBB);
        if(this->StreamEnabled == true){
          Builder.SetInsertPoint(ClonedConfigureCall);
          Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, ConstantInt::get(Int32Type, 1)}, "arrayidx");
          Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
          Value* LoadedStream = Builder.CreateLoad(IthStream);
          ClonedConfigureCall->setArgOperand(StreamArgIndex, LoadedStream);
        }

        Builder.SetInsertPoint(ClonedConfigureCall->getNextNode());
        Instruction* ConfgurationCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigureCall, One32bit));
        Instruction* NewBasicBlockFirstInstruction = SplitBlockAndInsertIfThen(ConfgurationCheck, ConfgurationCheck->getNextNode(), false);


        int NumberOfArgs = FunctionCall->getNumArgOperands() - 1; // Remove the output 

        std::vector<Value *> ArgsOfReplicationFunction;


        
        for(int ArgIndex = 0; ArgIndex < NumberOfArgs; ArgIndex++){
          std::vector<Instruction * > InstructionToClone;
          Value* Arg = FunctionCall->getArgOperand(ArgIndex);
          Value* OriginalArg = Arg;
          if(Instruction* ArgAsInstruction = dyn_cast<Instruction>(Arg)){
            Instruction* Clonelocation = NewBasicBlockFirstInstruction;
            bool Condition = dyn_cast_or_null<AllocaInst>(ArgAsInstruction) == nullptr;

            do{
              InstructionToClone.push_back(ArgAsInstruction);
              ArgAsInstruction = dyn_cast<Instruction>(ArgAsInstruction->getOperand(0));
              Condition = dyn_cast_or_null<AllocaInst>(ArgAsInstruction) == nullptr;
            } while (Condition);



            Instruction* PrevCloned = nullptr;
            for (unsigned Index = InstructionToClone.size(); Index-- > 0; ){
              Instruction* Cloned = InstructionToClone.at(Index)->clone();
              Cloned->insertBefore(Clonelocation);
              if(PrevCloned != nullptr){
                Cloned->setOperand(0, PrevCloned);
              }
                PrevCloned = Cloned;
            }
            
              OriginalArg = PrevCloned;
          }
          ArgsOfReplicationFunction.push_back(OriginalArg);
        }
        Builder.SetInsertPoint(NewBasicBlockFirstInstruction);

        Value* NewOutput = OutputObject->Replications[0];
        ArgsOfReplicationFunction.push_back(Builder.CreateLoad(NewOutput));
         
        
        Builder.CreateCall(FunctionToReplicate, ArgsOfReplicationFunction);
        
      
      
      Instruction* ClonedConfigure = PassAuxiliary->CudaConfigureCall->clone();
      ClonedConfigure->insertBefore(FirstInstructionOfNextBB);
      Builder.SetInsertPoint(ClonedConfigure->getNextNode());
      Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, PassAuxiliary->One32Bit));
      FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
      Builder.SetInsertPoint(FirstInstructionOfNextBB);

      Value* OrijinalOutput = OutputObject->OutputAllocation;
      Value* FirstReplicationOutput = OutputObject->Replications[0];
      Value* SizeOfOutput = OutputObject->MallocInstruction->getArgOperand(1);
      Value* CPUError = Errors.first;
      Value* GPUError = Errors.second;

      Value* LoadedOrijinalOutput = Builder.CreateLoad(OrijinalOutput);
      Value* LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
      Value* LoadedError = Builder.CreateLoad(GPUError);

      Builder.CreateCall(OutputObject->DetectionFunction, {LoadedOrijinalOutput, LoadedFirstReplicationOutput, SizeOfOutput, LoadedError});

      Builder.SetInsertPoint(FirstInstructionOfNextBB);

      Value* LoadedCPUerror = Builder.CreateLoad(CPUError);
      Value* BitcastedCPUError = Builder.CreateBitCast(LoadedCPUerror, PassAuxiliary->Int8PtrType);

      Value* LoadedGPUerror = Builder.CreateLoad(GPUError);
      Value* BitcastedGPUError = Builder.CreateBitCast(LoadedGPUerror, PassAuxiliary->Int8PtrType);

      Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedCPUError, BitcastedGPUError, PassAuxiliary->Four64Bit, PassAuxiliary->Two32Bit});

      LoadedCPUerror = Builder.CreateLoad(CPUError);
      Value* PointerToCPUError = Builder.CreateGEP(LoadedCPUerror, Zero32bit);
      Value* LoadedCPUErrorIndex = Builder.CreateLoad(PointerToCPUError);

      Instruction* ErrorCheck = dyn_cast<Instruction>(Builder.CreateICmpEQ(LoadedCPUErrorIndex, One32bit));
      Instruction* NewBBTerminator = SplitBlockAndInsertIfThen(ErrorCheck, ErrorCheck->getNextNode(),false);
      
      BranchInst* Br = dyn_cast<BranchInst>(ErrorCheck->getNextNode());
      Br->setOperand(1, dyn_cast<BasicBlock>(Br->getOperand(1))->getNextNode());
      Builder.SetInsertPoint(NewBBTerminator);
      LoadedCPUerror = Builder.CreateLoad(CPUError);
      PointerToCPUError = Builder.CreateGEP(LoadedCPUerror, Zero32bit);
      Builder.CreateStore(Zero32bit, PointerToCPUError);

      LoadedCPUerror = Builder.CreateLoad(CPUError);
      BitcastedCPUError = Builder.CreateBitCast(LoadedCPUerror, PassAuxiliary->Int8PtrType);

      LoadedGPUerror = Builder.CreateLoad(GPUError);
      BitcastedGPUError = Builder.CreateBitCast(LoadedGPUerror, PassAuxiliary->Int8PtrType);

      Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedGPUError, BitcastedCPUError,  PassAuxiliary->Four64Bit, PassAuxiliary->One32Bit});

      AllocaInst* OutputCPU = Builder.CreateAlloca(OutputObject->OutputType, 0, "Output_cpu");
      Instruction* OutputCpuAllocation = Builder.CreateCall(PassAuxiliary->MallocFunction, {OutputObject->MallocInstruction->getOperand(1)});
      Value* BitCastedOutputCPU = Builder.CreateBitCast(OutputCpuAllocation, OutputObject->OutputType);
      Builder.CreateStore(BitCastedOutputCPU, OutputCPU);

      Instruction* LoadedOutputCPU = Builder.CreateLoad(OutputCPU);
      BitCastedOutputCPU = Builder.CreateBitCast(OutputCpuAllocation, PassAuxiliary->Int8PtrType);
      LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
      Value* BitCastedFirstReplicationOutput = Builder.CreateBitCast(LoadedFirstReplicationOutput, PassAuxiliary->Int8PtrType);

      Instruction* CPUCpyCall = Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitCastedOutputCPU, BitCastedFirstReplicationOutput,  OutputObject->MallocInstruction->getOperand(1), PassAuxiliary->Two32Bit});

      createAndAllocateVariableAndreMemCpy(Builder, OutputObject, *CPUCpyCall->getNextNode(), PassAuxiliary->CudaMemCopy, IsLoop, 1);

        ClonedConfigureCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfigureCall->clone());
        ClonedConfigureCall->insertBefore(FirstInstructionOfNextBB);
        if(this->StreamEnabled == true){
          Builder.SetInsertPoint(ClonedConfigureCall);
          Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, ConstantInt::get(Int32Type, 2)}, "arrayidx");
          Builder.CreateCall(StreamCreateFunction, {IthStream, CudaStreamNonBlocking});
          Value* LoadedStream = Builder.CreateLoad(IthStream);
          ClonedConfigureCall->setArgOperand(StreamArgIndex, LoadedStream);
        }

        Builder.SetInsertPoint(ClonedConfigureCall->getNextNode());
        ConfgurationCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigureCall, One32bit));
        NewBasicBlockFirstInstruction = SplitBlockAndInsertIfThen(ConfgurationCheck, ConfgurationCheck->getNextNode(), false);


        NumberOfArgs = FunctionCall->getNumArgOperands() - 1; // Remove the output 

        ArgsOfReplicationFunction.clear();

        
        for(int ArgIndex = 0; ArgIndex < NumberOfArgs; ArgIndex++){
          std::vector<Instruction * > InstructionToClone;
          Value* Arg = FunctionCall->getArgOperand(ArgIndex);
          Value* OriginalArg = Arg;
          if(Instruction* ArgAsInstruction = dyn_cast<Instruction>(Arg)){
            Instruction* Clonelocation = NewBasicBlockFirstInstruction;
            bool Condition = dyn_cast_or_null<AllocaInst>(ArgAsInstruction) == nullptr;

            do{
              InstructionToClone.push_back(ArgAsInstruction);
              ArgAsInstruction = dyn_cast<Instruction>(ArgAsInstruction->getOperand(0));
              Condition = dyn_cast_or_null<AllocaInst>(ArgAsInstruction) == nullptr;
            } while (Condition);



            Instruction* PrevCloned = nullptr;
            for (unsigned Index = InstructionToClone.size(); Index-- > 0; ){
              Instruction* Cloned = InstructionToClone.at(Index)->clone();
              Cloned->insertBefore(Clonelocation);
              if(PrevCloned != nullptr){
                Cloned->setOperand(0, PrevCloned);
              }
                PrevCloned = Cloned;
            }
            
              OriginalArg = PrevCloned;
          }
          ArgsOfReplicationFunction.push_back(OriginalArg);
        }
        
        Builder.SetInsertPoint(NewBasicBlockFirstInstruction);

        NewOutput = OutputObject->Replications[1];
        ArgsOfReplicationFunction.push_back(Builder.CreateLoad(NewOutput));
        
        Builder.CreateCall(FunctionToReplicate, ArgsOfReplicationFunction);
  


        ClonedConfigure = PassAuxiliary->CudaConfigureCall->clone();
        ClonedConfigure->insertBefore(FirstInstructionOfNextBB);
        Builder.SetInsertPoint(ClonedConfigure->getNextNode());
        ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, PassAuxiliary->One32Bit));
        FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
        Builder.SetInsertPoint(FirstInstructionOfNextBB);

        OrijinalOutput = OutputObject->OutputAllocation;
        Value* SecondReplicationOutput = OutputObject->Replications[1];

        LoadedOrijinalOutput = Builder.CreateLoad(OrijinalOutput);
        Value* LoadedSecondReplicationOutput = Builder.CreateLoad(SecondReplicationOutput);
        LoadedError = Builder.CreateLoad(GPUError);

        Builder.CreateCall(OutputObject->DetectionFunction, {LoadedOrijinalOutput, LoadedSecondReplicationOutput, SizeOfOutput, LoadedError});

      Builder.SetInsertPoint(FirstInstructionOfNextBB);

      LoadedCPUerror = Builder.CreateLoad(CPUError);
      BitcastedCPUError = Builder.CreateBitCast(LoadedCPUerror, PassAuxiliary->Int8PtrType);

      LoadedGPUerror = Builder.CreateLoad(GPUError);
      BitcastedGPUError = Builder.CreateBitCast(LoadedGPUerror, PassAuxiliary->Int8PtrType);

      Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedCPUError, BitcastedGPUError, PassAuxiliary->Four64Bit, PassAuxiliary->Two32Bit});

      LoadedCPUerror = Builder.CreateLoad(CPUError);
      PointerToCPUError = Builder.CreateGEP(LoadedCPUerror, Zero32bit);
      LoadedCPUErrorIndex = Builder.CreateLoad(PointerToCPUError);

      ErrorCheck = dyn_cast<Instruction>(Builder.CreateICmpEQ(LoadedCPUErrorIndex, One32bit));
      NewBBTerminator = SplitBlockAndInsertIfThen(ErrorCheck, ErrorCheck->getNextNode(),false);
      

      Builder.SetInsertPoint(NewBBTerminator);
      LoadedOutputCPU = Builder.CreateLoad(OutputCPU);
      BitCastedOutputCPU = Builder.CreateBitCast(LoadedOutputCPU, PassAuxiliary->Int8PtrType);

      LoadedOrijinalOutput= Builder.CreateLoad(OrijinalOutput);
      Value* BitcastedOrijinalOutput = Builder.CreateBitCast(LoadedOrijinalOutput, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->CudaMemCopy, {BitcastedOrijinalOutput, BitCastedOutputCPU, SizeOfOutput, PassAuxiliary->One32Bit});

      Builder.SetInsertPoint(NewBBTerminator->getParent()->getNextNode()->getFirstNonPHI());

      LoadedOutputCPU = Builder.CreateLoad(OutputCPU);
      BitCastedOutputCPU = Builder.CreateBitCast(LoadedOutputCPU, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->FreeFunction, BitCastedOutputCPU);

      LoadedSecondReplicationOutput = Builder.CreateLoad(SecondReplicationOutput);
      Value* BitcastedSecondReplication = Builder.CreateBitCast(LoadedSecondReplicationOutput, PassAuxiliary->Int8PtrType);
      Instruction* LastCall = Builder.CreateCall(PassAuxiliary->CudaMemFree, BitcastedSecondReplication);

      Builder.SetInsertPoint(LastCall->getParent()->getNextNode()->getNextNode()->getFirstNonPHI());
      LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
      BitCastedFirstReplicationOutput = Builder.CreateBitCast(LoadedFirstReplicationOutput, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->CudaMemFree, BitCastedFirstReplicationOutput);

      LoadedGPUerror = Builder.CreateLoad(GPUError);
      BitcastedGPUError = Builder.CreateBitCast(LoadedGPUerror, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->CudaMemFree, BitcastedGPUError);

      LoadedCPUerror = Builder.CreateLoad(CPUError);
      BitcastedCPUError = Builder.CreateBitCast(LoadedCPUerror, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->FreeFunction, BitcastedCPUError);


      Instruction* LoadedCheckpoint = Builder.CreateLoad(CheckPoint);
      Value* BitcastedCheckpoint = Builder.CreateBitCast(LoadedCheckpoint, PassAuxiliary->Int8PtrType);
      Builder.CreateCall(PassAuxiliary->FreeFunction, BitcastedCheckpoint);










      










      

           
      

      




    return true;
    };
    

  
};