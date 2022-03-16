
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


       createAndAllocateVariableAndreMemCpy(Builder, OutputObject, LastInstructionOfPrevBB, PassAuxiliary->CudaMemCopy, IsLoop);


      for(int ReplicationIndex = 1; ReplicationIndex < NumberOfReplication; ReplicationIndex++){
        CallInst* ClonedConfigureCall = dyn_cast<CallInst>(PassAuxiliary->CudaConfigureCall->clone());
        ClonedConfigureCall->insertBefore(FirstInstructionOfNextBB);
        if(this->StreamEnabled == true){
          Builder.SetInsertPoint(ClonedConfigureCall);
          Value* IthStream = Builder.CreateInBoundsGEP(StreamArray, {Zero32bit, ConstantInt::get(Int32Type, ReplicationIndex)}, "arrayidx");
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
            errs() << InstructionToClone.size() << "\n";
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

        Value* NewOutput = OutputObject->Replications[ReplicationIndex - 1];
        ArgsOfReplicationFunction.push_back(Builder.CreateLoad(NewOutput));

        errs() << *Builder.CreateCall(FunctionToReplicate, ArgsOfReplicationFunction) << "\n";
        
      }

      Instruction* ClonedConfigure = PassAuxiliary->CudaConfigureCall->clone();
      ClonedConfigure->insertAfter(FirstInstructionOfNextBB);
      Builder.SetInsertPoint(ClonedConfigure->getNextNode());
      Instruction* ConfigureCheck = dyn_cast<Instruction>(Builder.CreateICmpNE(ClonedConfigure, PassAuxiliary->One32Bit));
      FirstInstructionOfNextBB = SplitBlockAndInsertIfThen(ConfigureCheck, ConfigureCheck->getNextNode(), false);
      Builder.SetInsertPoint(FirstInstructionOfNextBB);

      Value* OrijinalOutput = OutputObject->OutputAllocation;
      Value* FirstReplicationOutput = OutputObject->Replications[0];
      Value* SecondReplicationOutput = OutputObject->Replications[1];
      Value* SizeOfOutput = OutputObject->MallocInstruction->getArgOperand(1);

      Value* LoadedOrijinalOutput = Builder.CreateLoad(OrijinalOutput);
      Value* LoadedFirstReplicationOutput = Builder.CreateLoad(FirstReplicationOutput);
      Value* LoadedSecondReplicationOutput = Builder.CreateLoad(SecondReplicationOutput);

      errs() << *OutputObject->MajorityVotingFunction->getFunctionType() << "\n";
      errs() << *LoadedOrijinalOutput->getType() << "\n";
      errs() << *LoadedFirstReplicationOutput->getType() << "\n";
      errs() << *LoadedSecondReplicationOutput->getType() << "\n";
      errs() << *SizeOfOutput->getType() << "\n";

      Builder.CreateCall(OutputObject->MajorityVotingFunction, {LoadedOrijinalOutput, LoadedFirstReplicationOutput, LoadedSecondReplicationOutput, SizeOfOutput});



      

      




    return false;
    };
    

  
};