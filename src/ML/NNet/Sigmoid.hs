{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

module ML.NNet.Sigmoid where

import Control.Monad.IO.Class (MonadIO (liftIO))
import Data.BlasM
import Data.Proxy
import Data.Serialize
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random
import Text.Printf

euler :: Double
euler = 2.7182818284

type SigmoidIn mx w h d = Matrix mx w h d

sigSerialize :: Monad m => conf -> (Get (m ()), () -> m Put)
sigSerialize _ =
  ( pure $ pure (),
    const $ pure $ pure ()
  )

sigForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> Matrix mx w h d -> m (Matrix mx w h d, SigmoidIn mx w h d)
sigForward _ mx = do
  r <- mx `applyFunction` Div (Const 1.0) (Add (Const 1.0) (Exp (Const euler) (Mul (Const (-1.0)) Value)))
  pure (r, r)

sigBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => () -> SigmoidIn mx w h d -> Matrix mx w h d -> m (Matrix mx w h d, ())
sigBackward _ sig dz = do
  -- sigmoid(x) * (1 - sigmoid(x))
  e <- applyFunction sig (Mul Value (Sub (Const 1.0) Value))
  dy <- mult e dz
  -- dzr <- mxToLists dz
  -- dyr <- mxToLists dy
  -- liftIO $ printf "-- sigback --\n%s\n%s\n" (show dzr) (show dyr)
  pure (dy, ())

sigAvg :: Monad m => (() -> () -> m (), () -> Int -> m ())
sigAvg = (const (const (pure ())), const (const (pure ())))

sigUpd :: (Monad m) => conf -> () -> () -> m ()
sigUpd _ _ _ = pure ()

sigmoid :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => Proxy '(w, h, d) -> Layer m mx () (SigmoidIn mx w h d) () w h d w h d conf mod g
sigmoid _ = Layer sigForward sigBackward sigAvg sigUpd sigmoidInit sigSerialize

sigmoidInit :: (Monad m, RandomGen g) => WeightInitializer g -> g -> m ((), g)
sigmoidInit _ g = do
  pure ((), g)
