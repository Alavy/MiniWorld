using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class GameEvents 
{
    public static Action<BlockType> OnChooseObjectChangedEvent;
    public static void OnChooseObjectChangedCalled(BlockType type) => OnChooseObjectChangedEvent?.Invoke(type);

    public static Action<GameMode> OnChooseModechangedEvent;
    public static void OnChooseModeChangedCalled(GameMode type) => OnChooseModechangedEvent?.Invoke(type);

    public static Action OnCoverUIEnterEvent;
    public static void OnCoverUIEnterCalled() => OnCoverUIEnterEvent?.Invoke();

    public static Action OnCoverUIExitEvent;
    public static void OnCoverUIExitCalled() => OnCoverUIExitEvent?.Invoke();
}
