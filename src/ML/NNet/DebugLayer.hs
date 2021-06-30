{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ML.NNet.DebugLayer where

import Control.Monad (when)
import Control.Monad.IO.Class (MonadIO (..))
import Data.BlasM
import Data.Proxy
import Data.Serialize
import Data.Text
import GHC.TypeLits
import ML.NNet
import ML.NNet.Init.RandomFun
import System.Random

data DebugSt m = DebugSt String Bool Bool (Text -> m ())

debugSerialize :: Monad m => conf -> (Get (m (DebugSt m)), DebugSt m -> m Put)
debugSerialize _ =
  ( pure $ pure (DebugSt "oops you left debug on" False False (const $ pure ())),
    const $ pure $ pure ()
  )

debugForward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => DebugSt m -> Matrix mx w h d -> m (Matrix mx w h d, ())
debugForward (DebugSt s True _ logAct) mx = do
  csv <- mxToCSV mx
  -- csv <- avgMxs [mx]
  -- when (csv > 10 || csv < -10) (logAct (pack $ s <> " forward: " <> show csv))
  logAct (pack (s <> " forward\n" <> show csv))
  pure (mx, ())
debugForward _ mx = pure (mx, ())

debugBackward :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx) => DebugSt m -> () -> Matrix mx w h d -> m (Matrix mx w h d, ())
debugBackward (DebugSt s _ True logAct) _ mx = do
  csv <- avgMxs [mx]
  -- when (csv > 10 || csv < -10) (logAct (pack $ s <> " backward: " <> show csv))
  logAct (pack (s <> " backward\n" <> show csv))
  pure (mx, ())
debugBackward _ _ mx = pure (mx, ())

debugG :: (Monad m, MonadIO m) => (() -> () -> m (), () -> Int -> m ())
debugG = (const (const (liftIO (putStrLn "gradl") >> pure ())), const (const (liftIO (putStrLn "gradl") >> pure ())))

debugU :: (Monad m) => conf -> DebugSt m -> () -> m (DebugSt m)
debugU _ st@(DebugSt lbl _ _ act) _ = act (pack $ lbl <> ":update") >> pure st

debugLayer :: (KnownNat w, KnownNat h, KnownNat d, BlasM m mx, RandomGen g) => String -> Bool -> Bool -> (Text -> m ()) -> Layer m mx (DebugSt m) () () w h d w h d conf mod g
debugLayer lbl logForward logBackward logAction = Layer debugForward debugBackward debugG debugU (debugInit lbl logForward logBackward logAction) debugSerialize

debugInit :: (BlasM m mx, RandomGen g) => String -> Bool -> Bool -> (Text -> m ()) -> WeightInitializer g -> g -> m (DebugSt m, g)
debugInit lbl logForward logBackward logAction _ gen = do
  pure (DebugSt lbl logForward logBackward logAction, gen)
