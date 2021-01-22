{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.Matrix.CLBlas where

import Foreign.C.Types
import Foreign.Ptr
import Data.Int
import Data.Matrix.OpenCL hiding (failLeft,clTry)
import Data.Word
import Foreign.ForeignPtr
import Foreign.Storable
import Foreign.Marshal.Alloc
import Data.Word
import Data.Vector.Storable
import Control.Monad.IO.Class
import Data.Proxy
import Debug.Trace

#include <clblashs.h> 
#include <openclhs.h>

newtype CLBlasStatusCode = CLBlasStatusCode CInt
    deriving (Eq,Show)
#{enum CLBlasStatusCode, CLBlasStatusCode, clblasSuccess,
    clblasInvalidValue,
    clblasInvalidCommandQueue,
    clblasInvalidContext,
    clblasInvalidMemObject,
    clblasInvalidDevice,
    clblasInvalidEventWaitList,
    clblasOutOfResources,
    clblasOutOfHostMemory,
    clblasInvalidOperation,
    clblasCompilerNotAvailable,
    clblasBuildProgramFailure,
    clblasNotImplemented,
    clblasNotInitialized,
    clblasInvalidMatA,
    clblasInvalidMatB,
    clblasInvalidMatC,
    clblasInvalidVecX,
    clblasInvalidVecY,
    clblasInvalidDim,
    clblasInvalidLeadDimA,
    clblasInvalidLeadDimB,
    clblasInvalidLeadDimC,
    clblasInvalidIncX,
    clblasInvalidIncY,
    clblasInsufficientMemMatA,
    clblasInsufficientMemMatB,
    clblasInsufficientMemMatC,
    clblasInsufficientMemVecX,
    clblasInsufficientMemVecY}

newtype CLBlasTranspose = CLBlasTranspose CInt deriving (Eq,Show)
#{enum CLBlasTranspose,CLBlasTranspose,clblasNoTrans,clblasTrans,clblasConjTrans}

newtype CLBlasOrder = CLBlasOrder CInt deriving (Eq,Show)
#{enum CLBlasOrder,CLBlasOrder,clblasRowMajor,clblasColumnMajor}

type SizeT = #type size_t



class (Storable a, Floating a) => CLBlasType a where
  clblasTgemm :: CLBlasOrder -> CLBlasTranspose -> CLBlasTranspose -> SizeT -> SizeT -> SizeT -> a -> CLMem_ -> SizeT -> SizeT -> CLMem_ -> SizeT -> SizeT -> a -> CLMem_ -> SizeT -> SizeT -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode
  clblasTaxpy :: SizeT -> a -> CLMem_ -> SizeT -> CInt -> CLMem_ -> SizeT -> CInt -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode
  clblasTDot :: Proxy a -> SizeT -> CLMem_ -> SizeT -> CLMem_ -> SizeT -> CInt -> CLMem_ -> SizeT -> CInt -> CLMem_ -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode 
  clblasTCopy :: Proxy a -> SizeT -> CLMem_ -> SizeT -> CInt -> CLMem_ -> SizeT -> CInt -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode
  clblasTypeToDouble :: a -> Double
  clblasTypeCStringRep :: a -> String
  clblasTypeNearZero :: Proxy a -> Double

instance CLBlasType CDouble where
  clblasTgemm = clblasDgemm_
  clblasTaxpy = clblasDaxpy_
  clblasTDot _ = clblasDdot_
  clblasTCopy _ = clblasDcopy_
  clblasTypeToDouble = fromRational . toRational
  clblasTypeCStringRep _ = "double"
  clblasTypeNearZero _ = encodeFloat (signif+1) expo - 1.0
                      where (signif,expo) = decodeFloat (1.0::CDouble)

instance CLBlasType CFloat where
  clblasTgemm = clblasSgemm_
  clblasTaxpy = clblasSaxpy_
  clblasTDot _ = clblasSdot_
  clblasTCopy _ = clblasScopy_
  clblasTypeToDouble = fromRational . toRational
  clblasTypeCStringRep _ = "float"
  clblasTypeNearZero _ = encodeFloat (signif+1) expo - 1.0
                      where (signif,expo) = decodeFloat (1.0::CFloat)


failLeft :: (MonadFail m, MonadIO m) => IO (Either String a) -> m a
failLeft ie = do
  e <- liftIO ie
  case e of
    Left e' -> fail e'
    Right v -> pure v

-- TODO add a bracket here
clblasTry :: (MonadIO m, MonadFail m) => IO CLBlasStatusCode -> IO a -> m (Either String a) 
clblasTry st f = do
  r <- liftIO st
  if r /= clblassuccess
    then pure (Left (show r))
    else liftIO $ Right <$> f

-- | clblasGemm - dense matrix multiplication
clblasGemm :: (MonadFail m, MonadIO m, CLBlasType a) => CLBlasOrder -> CLBlasTranspose -> CLBlasTranspose 
  -> Word64 -- ^ number of rows in first matrix
  -> Word64  -- ^ number of columns in second matrix
  -> Word64  -- ^ number of cols of first matrix and rows of second matrix
  -> a -- ^ scale first matrix by const
  -> CLMem a -- ^ first matrix memory
  -> Word64 -- ^  First matrix offset to start reading from
  -> Word64  -- ^ First matrix first dimension (rows for rowmajor, cols for colmajor)
  -> CLMem a -- ^ second matrix memory
  -> Word64 -- ^  Second matrix offset to start reading from
  -> Word64 -- ^ Second matrix first dimension (rows for rowmajor, cols for colmajor)
  -> a -- ^ constant to add to output matrix
  -> CLMem a -- ^ output matrix memory 
  -> Word64 -- ^ output matrix offset size
  -> Word64 -- ^ output matrix first dimension (rows for rowmajor, cols for colmajor)
  -> CLCommandQueue -> m CLEvent
clblasGemm layout transA transB m n k alpha (CLMem al a ap) aoff lda (CLMem bl b bp) boff ldb beta (CLMem cl c cp) coff ldc queue =
  failLeft $
    alloca $ \e' ->
      clblasTry 
        (withForeignPtr a (\a' ->
            withForeignPtr b (\b' -> 
              withForeignPtr c (\c' -> 
                withForeignPtr queue (\q' -> do
                    poke e' nullPtr
                    -- transA = clblasNoTrans
                    -- LDA < k if not transposed? -- pass m,k,a,offa,lda
                    --putStrLn $ "transA: " <> show transA <> " transB: " <> show transB <> " (m,n,k):" <> show (m,n,k) <> " (lda,ldb,ldc):" <> show (lda,ldb,ldc)
                    r <- clblasTgemm layout transA transB m n k alpha a' aoff lda b' boff ldb beta c' coff ldc 1 q' 0 nullPtr e'
                    --putStrLn $ "Tgemm fin with: " <> show r
                    pure r
                ) 
              )
            )
          )
        )
        (peek e' >>= clEventPtrToForeignPtr [a,b,c])

-- Vector * constant + Vector, overwrites the second vector
-- in future to avoid overwrite we copy the second vector to a new or existing third vector
clblasAxpy :: (MonadFail m, MonadIO m, CLBlasType a) => CLCommandQueue -> CLMem a -> a -> CLMem a -> m CLEvent 
clblasAxpy queue (CLMem len1 fpclmem1 prxy1) alpha (CLMem len2 fpclmem2 _)
  | len1 /= len2 = traceStack "add wrong" $ fail $ "Cannot add two vectors of different sizes " <> show (len1,len2)
  | otherwise =
    failLeft $ 
      alloca $ \e -> 
                  clblasTry
                    (withForeignPtr queue (\queue' ->  
                        (withForeignPtr fpclmem1 (\mem1 -> 
                            withForeignPtr fpclmem2 (\mem2 -> do
                                poke e nullPtr
                                clblasTaxpy (fromIntegral len1) alpha mem1 0 1 mem2 0 1 1 queue' 0 nullPtr e
                              )
                          )
                        )
                      )
                    )
                    (peek e >>= clEventPtrToForeignPtr [fpclmem1,fpclmem2])

clblasDot :: (MonadFail m, MonadIO m, CLBlasType a) => CLContext -> CLCommandQueue -> CLMem a -> CLMem a -> CLMem a -> m CLEvent 
clblasDot ctx queue (CLMem sz1 fpclmem1 _) (CLMem sz2 fpclmem2 _) (CLMem sz3 fpclmem3 pxy)
  | sz1 /= sz2 = fail "Cannot dot two vectors of different sizes"
  | sz3 < 1 = fail "Must have at least one slot in result memory slot"
  | otherwise =
    failLeft $ 
      alloca $ \e -> do
                  (CLMem _ scratch _) <- clCreateBuffer ctx clMemReadWrite sz1 pxy
                  clblasTry
                    (withForeignPtr queue (\queue' ->  
                        (withForeignPtr fpclmem1 (\mem1 -> 
                            withForeignPtr fpclmem2 (\mem2 -> 
                              withForeignPtr fpclmem3 (\mem3 -> do
                                poke e nullPtr
                                -- needs a scratchbuff...y
                                withForeignPtr scratch (\scratch' -> 
                                  clblasTDot pxy (fromIntegral sz1) mem3 0 mem1 0 1 mem2 0 1 scratch' 1 queue' 0 nullPtr e -- if this fails, need to call release on e

                                  )
                              )
                            )
                          )
                        )
                      )
                    )
                    (peek e >>= clEventPtrToForeignPtr [fpclmem1,fpclmem2,fpclmem3,scratch])
     

clblasClone ::  (MonadFail m, MonadIO m, CLBlasType a) => CLContext -> CLCommandQueue -> CLMem a -> m (CLMem a)
clblasClone ctx queue (CLMem sz fpmem pxy) = do
  (CLMem sz2 fpmem2 _) <- clCreateBuffer ctx clMemReadWrite sz pxy 
  --liftIO $ putStrLn $ "cloning sz:" <> show sz <> " into sz2: " <> show sz2
  failLeft $
    clblasTry
      (alloca $ \e -> 
        (withForeignPtr queue (\queue' -> 
          (withForeignPtr fpmem (\mem -> 
            (withForeignPtr fpmem2 (\mem2 -> 
              do poke e nullPtr
                 r <- clblasTCopy pxy (fromIntegral sz) mem 0 1 mem2 0 1 1 queue' 0 nullPtr e
                 if r == clblassuccess 
                 then peek e >>= clEventPtrToForeignPtr [fpmem,fpmem2] >>= clWaitForEvent >> pure r
                 else pure r
              )
            )
          )
          )
        )
        )
      )
      (
        pure (CLMem sz2 fpmem2 pxy)
      )


foreign import ccall "clblasSaxpy" clblasSaxpy_ :: SizeT -> CFloat
    -> CLMem_ -> SizeT -> CInt
    -> CLMem_ -> SizeT -> CInt
    -> CLUInt -> Ptr CLCommandQueue_ 
    -> CLUInt -> Ptr CLEvent_
    -> Ptr CLEvent_
    -> IO CLBlasStatusCode

foreign import ccall "clblasDaxpy" clblasDaxpy_ :: SizeT -> CDouble
    -> CLMem_ -> SizeT -> CInt
    -> CLMem_ -> SizeT -> CInt
    -> CLUInt -> Ptr CLCommandQueue_ 
    -> CLUInt -> Ptr CLEvent_
    -> Ptr CLEvent_
    -> IO CLBlasStatusCode

-- alpha * A * x + beta * y
foreign import ccall "clblasDgemm" clblasDgemm_ :: CLBlasOrder -> CLBlasTranspose -> CLBlasTranspose
    -> SizeT -> SizeT -> SizeT -- Matrix A rows, Matrix A columns, 
    -> CDouble -- alpha (scalar alpha)
    -> CLMem_ -> SizeT -> SizeT -- mem, a offset, a ld
    -> CLMem_ -> SizeT -> SizeT -- mem b, b offset, b ld
    -> CDouble -- beta
    -> CLMem_ -> SizeT -> SizeT -- out buf, buf offset, buff ld
    -> CLUInt -> Ptr CLCommandQueue_ 
    -> CLUInt -> Ptr CLEvent_
    -> Ptr CLEvent_
    -> IO CLBlasStatusCode

foreign import ccall "clblasSgemm" clblasSgemm_ :: CLBlasOrder -> CLBlasTranspose -> CLBlasTranspose
    -> SizeT -> SizeT -> SizeT -- Matrix A rows, Matrix A columns, 
    -> CFloat -- alpha (scalar alpha)
    -> CLMem_ -> SizeT -> SizeT -- mem, a offset, a ld
    -> CLMem_ -> SizeT -> SizeT -- mem b, b offset, b ld
    -> CFloat -- beta
    -> CLMem_ -> SizeT -> SizeT -- out buf, buf offset, buff ld
    -> CLUInt -> Ptr CLCommandQueue_ 
    -> CLUInt -> Ptr CLEvent_
    -> Ptr CLEvent_
    -> IO CLBlasStatusCode

foreign import ccall "clblasDdot" clblasDdot_ :: SizeT -> CLMem_ -> SizeT 
  -> CLMem_ -> SizeT -> CInt
  -> CLMem_ -> SizeT -> CInt
  -> CLMem_ -- scratch buf
  -> CLUInt -> Ptr CLCommandQueue_ 
  -> CLUInt -> Ptr CLEvent_
  -> Ptr CLEvent_
  -> IO CLBlasStatusCode

foreign import ccall "clblasSdot" clblasSdot_ :: SizeT -> CLMem_ -> SizeT 
  -> CLMem_ -> SizeT -> CInt
  -> CLMem_ -> SizeT -> CInt
  -> CLMem_ -- scratch buf
  -> CLUInt -> Ptr CLCommandQueue_ 
  -> CLUInt -> Ptr CLEvent_
  -> Ptr CLEvent_
  -> IO CLBlasStatusCode

foreign import ccall "clblasScopy" clblasScopy_ :: SizeT -> CLMem_ -> SizeT -> CInt -> CLMem_ -> SizeT -> CInt -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode
foreign import ccall "clblasDcopy" clblasDcopy_ :: SizeT -> CLMem_ -> SizeT -> CInt -> CLMem_ -> SizeT -> CInt -> CLUInt -> Ptr CLCommandQueue_ -> CLUInt -> Ptr CLEvent_ -> Ptr CLEvent_ -> IO CLBlasStatusCode

foreign import ccall "clblasSetup" clblasSetup :: IO CLBlasStatusCode
foreign import ccall "clblasTeardown" clblasTeardown :: IO ()
