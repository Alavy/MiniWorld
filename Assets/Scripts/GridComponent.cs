using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class GridComponent : MonoBehaviour
{
    [SerializeField]
    private GameObject red;
    [SerializeField]
    private GameObject blue;
    [SerializeField]
    private float vanishTime = 0.2f;
    [SerializeField]
    private float popUpTime = 0.3f;
    [SerializeField]
    private Color selectColor=Color.black;
    [SerializeField]
    private GameObject parent;
    [SerializeField]
    private MeshRenderer gridTile;

    private Color m_defaultGridColor = Color.white;
    private Color m_selectGridColor = Color.green;
    private Color m_pathGridColor = Color.magenta;

    private BlockType m_blockType = BlockType.None;
    private GridType m_gridType = GridType.None;

    private Color m_redColor;
    private Color m_blueColor;

    private void Start()
    {
        red.SetActive(false);
        blue.SetActive(false);
        m_blockType = BlockType.None;
        m_gridType = GridType.None;

        m_redColor = red.GetComponent<MeshRenderer>().material.color;
        m_blueColor = blue.GetComponent<MeshRenderer>().material.color;
    }
    public BlockType GetBlockType()
    {
        return m_blockType;
    }
    public void HoverBlock(BlockType type)
    {


        if (m_blockType!=BlockType.None)
        {
            if (m_blockType != type)
                return;

            if (m_blockType == BlockType.Red)
            {
                red.GetComponent<MeshRenderer>().material.color = selectColor;
            }
            else if (m_blockType == BlockType.Blue)
            {
                blue.GetComponent<MeshRenderer>().material.color = selectColor;
            }
            return;
        }
        if (type == BlockType.Red)
        {
            red.GetComponent<MeshRenderer>().material.color = m_redColor;

            red.SetActive(true);
            blue.SetActive(false);
        }
        else if (type == BlockType.Blue)
        {
            red.SetActive(false);
            blue.SetActive(true);

            blue.GetComponent<MeshRenderer>().material.color = m_blueColor;
        }
    }
    public Mesh GetActiveMesh()
    {
        if (m_blockType == BlockType.Red)
        {
            return red.GetComponent<MeshFilter>().sharedMesh;
        }else if(m_blockType== BlockType.Blue)
        {
            return blue.GetComponent<MeshFilter>().sharedMesh;
        }
        return null;
    }
    public void UnHoverBlock()
    {
        if (m_blockType == BlockType.None)
        {
            red.SetActive(false);
            blue.SetActive(false);
        }
        else
        {
            if (m_blockType == BlockType.Red)
            {
                red.GetComponent<MeshRenderer>().material.color = m_redColor;
                red.SetActive(true);
                blue.SetActive(false);
            }
            else if (m_blockType == BlockType.Blue)
            {
                red.SetActive(false);
                blue.SetActive(true);
                blue.GetComponent<MeshRenderer>().material.color = m_blueColor;

            }
        }
  
    }
    
    public void SetBlockType(BlockType type)
    {
        if (m_blockType != BlockType.None)
            return;

        m_blockType = type;

        if (type == BlockType.Red)
        {
            red.SetActive(true);

            red.transform.localScale = Vector3.zero;
            red.LeanScale(Vector3.one, popUpTime);

            blue.SetActive(false);
        }
        else if (type == BlockType.Blue)
        {
            red.SetActive(false);
            blue.SetActive(true);

            blue.transform.localScale = Vector3.zero;
            blue.LeanScale(Vector3.one, popUpTime);
        }
    }
    public void VanishBlock(BlockType type)
    {
        if (m_blockType != type)
            return;

        if (m_blockType == BlockType.Red)
        {
            red.LeanScale(Vector3.zero, vanishTime).setOnComplete(() =>
            {
                red.SetActive(false);
                red.transform.localScale = Vector3.one;
            });
        }else if(m_blockType == BlockType.Blue)
        {
            blue.LeanScale(Vector3.zero, vanishTime).setOnComplete(() =>
            {
                blue.SetActive(false);
                blue.transform.localScale = Vector3.one;
            });
        }
        m_blockType = BlockType.None;
    }
    public GameObject ReturnParent()
    {
        return parent;
    }
    public MeshRenderer ReturnGridTile()
    {
        return gridTile;
    }
    public void SelectGridTile()
    {
        m_gridType = GridType.Selected;
        gridTile.material.color = m_selectGridColor;
    }
    public void HoverGridTile()
    {
        if(m_gridType==GridType.None)
        {
            gridTile.material.color = m_selectGridColor;
        }

    }
    public void UnHoverGridTile()
    {
        if (m_gridType == GridType.None)
        {
            gridTile.material.color = m_defaultGridColor;

        }
    }
    public void PaintGridTile()
    {
        m_gridType = GridType.Painted;
        gridTile.material.color = m_pathGridColor;

    }
    public void ClearGridTile()
    {
        m_gridType = GridType.None;
        gridTile.material.color = m_defaultGridColor;
    }
}
