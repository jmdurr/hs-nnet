{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module ML.NNet.Tanh where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

type TanV mx w h d = Matrix mx w h d

tanF :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> Matrix mx w h d -> m (Matrix mx w h d, TanV mx w h d)
tanF _ mx = do
  r <- mx `applyFunction` (Tanh Value)
  pure (r, r)

tanB :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> TanV mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, ())
tanB _ tv dz = do
  -- sigmoid(x) * (1 - sigmoid(x))
  e <- applyFunction tv (Sub (Const 1.0) (Mul Value Value))
  dy <- mult e dz
  pure (dy, ())

tanAvg :: Monad m => [()] -> m ()
tanAvg _ = pure ()

tanUpd :: (Monad m) => conf -> () -> () -> m ()
tanUpd _ _ _ = pure ()

tanH :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Proxy '(w, h, d) -> Layer m mx () (TanV mx w h d) () w h d w h d conf mod g
tanH _ = Layer tanF tanB tanAvg tanUpd tanInit

tanInit :: (Monad m, RandomGen g) => WeightInitializer g -> g -> m ((), g)
tanInit _ g = pure ((), g)
