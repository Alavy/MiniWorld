using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIController : MonoBehaviour
{
    [SerializeField]
    private GameObject selectObjblockUI;
    [SerializeField]
    private GameObject selectPathblockUI;
    private void Start()
    {
        selectObjblockUI.SetActive(false);
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
            selectObjblockUI.SetActive(false);
            selectPathblockUI.SetActive(true);
            GameEvents.OnChooseModeChangedCalled(GameMode.Builder);

        }
        else if (order == 1)
        {
            GameEvents.OnChooseModeChangedCalled(GameMode.NonBuilder);
            selectObjblockUI.SetActive(true);
            selectPathblockUI.SetActive(false);

        }

    }
    public void ChoosePathType(int order)
    {
        if (order == 0)
        {
            GameEvents.OnChoosePathTypeChangedCalled(PathType.Straight);

        }
        else if (order == 1)
        {
            GameEvents.OnChoosePathTypeChangedCalled(PathType.Diagonal);
        }
    }
    public void CoverUIEnter()
    {
        //Debug.Log("Enter");
        GameEvents.OnCoverUIEnterCalled();
    }
}
