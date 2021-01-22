{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module ML.NNet.Sigmoid where

import Control.Monad.IO.Class
import Data.BlasM
import Data.Proxy
import GHC.TypeLits
import ML.NNet
import System.Random

euler :: Double
euler = 2.7182818284

type SigmoidIn mx w h d = Matrix mx w h d

sigForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> Matrix mx w h d -> m (Matrix mx w h d, SigmoidIn mx w h d)
sigForward _ mx = do
  r <- mx `applyFunction` Div (Const 1.0) (Add (Const 1.0) (Exp (Const euler) (Mul (Const (-1.0)) Value)))
  pure (r, mx)

sigBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> SigmoidIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, ())
sigBackward _ sig dz = do
  -- sigmoid(x) * (1 - sigmoid(x))
  e <- applyFunction sig (Sub (Const 1.0) Value)
  g <- add sig e
  dy <- mult g dz
  pure (dy, ())

sigAvg :: Monad m => [()] -> m ()
sigAvg _ = pure ()

sigUpd :: (Monad m) => conf -> () -> () -> m ()
sigUpd _ _ _ = pure ()

sigmoid :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Proxy '(w, h, d) -> Layer m mx () (SigmoidIn mx w h d) () w h d w h d conf mod g
sigmoid _ = Layer sigForward sigBackward sigAvg sigUpd sigmoidInit

sigmoidInit :: (Monad m, RandomGen g) => (g -> (Double, g)) -> g -> m ((), g)
sigmoidInit _ g = do
  pure ((), g)
