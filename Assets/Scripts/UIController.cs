using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIController : MonoBehaviour
{
    [SerializeField]
    private GameObject blockUI;
    private void Start()
    {
        blockUI.SetActive(false);
    }
    public void ChooseObject(int order)
    {
        if (order == 0)
        {
            GameEvents.OnChooseObjectChangedCalled(BlockType.Red);

        }
        else if(order==1)
        {
            GameEvents.OnChooseObjectChangedCalled(BlockType.Blue);

        }
    }
    public void ChooseMode(int order)
    {
        if (order == 0)
        {
            blockUI.SetActive(false);
            GameEvents.OnChooseModeChangedCalled(GameMode.Builder);

        }
        else if (order == 1)
        {
            GameEvents.OnChooseModeChangedCalled(GameMode.NonBuilder);
            blockUI.SetActive(true);
        }

    }
    public void CoverUIEnter()
    {
        //Debug.Log("Enter");
        GameEvents.OnCoverUIEnterCalled();
    }
}
