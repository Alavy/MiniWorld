using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum GameMode
{
    Builder,
    NonBuilder
}
public enum BlockType
{
    Blue,
    Red,
    None
}
public class GridManager : MonoBehaviour
{
    [SerializeField]
    private Vector2 gridSize = new Vector2(20,10);
    [SerializeField]
    private Vector2 gridDim = new Vector2(1, 1);
    [SerializeField]
    private GameObject gridTile;
    [SerializeField]
    private LayerMask blockLayer;

    [SerializeField]
    private GameObject redCombineMeshHolder;
    [SerializeField]
    private GameObject blueCombineMeshHolder;
    [SerializeField]
    private float moveSpeed = 20f;

    private Camera m_cam;

    private GridComponent m_currentSelectedComponent;

    private BlockType m_spawnBlockType = BlockType.Red;
    private GameMode m_gameMode = GameMode.Builder;

    private Vector3 m_prevMouse;
    private Dictionary<Transform, GridComponent> m_gridElements;
    private bool m_disableInput = false;

    private Vector3 m_pastMouse;
    List<CombineInstance> m_redCombineMeshes;
    List<CombineInstance> m_blueCombineMeshes;

    private void Awake()
    {
        m_cam = Camera.main;
        m_gridElements = new Dictionary<Transform, GridComponent>();
        m_redCombineMeshes = new List<CombineInstance>();
        m_blueCombineMeshes = new List<CombineInstance>();

        generateGrid();

    }
    private void OnEnable()
    {
        GameEvents.OnChooseObjectChangedEvent += onChooseObjectChanged;
        GameEvents.OnChooseModechangedEvent += onGameModeChanged;
        GameEvents.OnCoverUIEnterEvent += onCoverUIEnter;
        //GameEvents.OnCoverUIExitEvent += onCoverUIExit;
    }
    private void onCoverUIEnter()
    {
        m_disableInput = true;
    }
   
    private void onChooseObjectChanged(BlockType type)
    {
        m_spawnBlockType = type;
    }
    private void onGameModeChanged(GameMode gameMode)
    {
        if (gameMode == GameMode.NonBuilder)
        {


            foreach (var item in m_gridElements.Values)
            {
                if (item.GetBlockType() == BlockType.Red)
                {
                    CombineInstance combine = new CombineInstance();
                    combine.mesh = item.GetActiveMesh();
                    combine.transform = item.transform.localToWorldMatrix;

                    m_redCombineMeshes.Add(combine);

                }else if (item.GetBlockType() == BlockType.Blue)
                {
                    CombineInstance combine = new CombineInstance();
                    combine.mesh = item.GetActiveMesh();
                    combine.transform = item.transform.localToWorldMatrix;

                    m_blueCombineMeshes.Add(combine);
                }
            }

            Mesh redMesh = new Mesh();
            redMesh.CombineMeshes(m_redCombineMeshes.ToArray());
            redCombineMeshHolder.GetComponent<MeshFilter>().sharedMesh = redMesh;

            Mesh blueMesh = new Mesh();
            blueMesh.CombineMeshes(m_blueCombineMeshes.ToArray());
            blueCombineMeshHolder.GetComponent<MeshFilter>().sharedMesh = blueMesh;

            redCombineMeshHolder.SetActive(true);
            blueCombineMeshHolder.SetActive(true);
            m_redCombineMeshes.Clear();
            m_blueCombineMeshes.Clear();

            foreach (var item in m_gridElements.Values)
            {
                item.gameObject.SetActive(false);
            }
        }
        else if (gameMode == GameMode.Builder)
        {
            foreach (var item in m_gridElements.Values)
            {
                item.gameObject.SetActive(true);
            }
            redCombineMeshHolder.SetActive(false);
            blueCombineMeshHolder.SetActive(false);

        }
    }
    private void OnDisable()
    {
        GameEvents.OnChooseObjectChangedEvent -= onChooseObjectChanged;
        GameEvents.OnChooseModechangedEvent -= onGameModeChanged;
        GameEvents.OnCoverUIEnterEvent -= onCoverUIEnter;
        //GameEvents.OnCoverUIExitEvent -= onCoverUIExit;
    }

    private void checkBlock()
    {
        Ray ray = m_cam.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, 1000, blockLayer))
        {
            GridComponent info;
            m_disableInput = false;

            if (m_gridElements.TryGetValue(hit.transform,out info))
            {
                if(info != m_currentSelectedComponent)
                {
                    if (m_currentSelectedComponent != null)
                    {
                        m_currentSelectedComponent.UnHoverBlock();
                    }
                    info.HoverBlock(m_spawnBlockType);
                    m_currentSelectedComponent = info;
                }
                

            }
        }
    }
    private void Start()
    {

    }
    private void generateGrid()
    {
        for (int z = 0; z < gridSize.y; z++)
        {
            for (int x = 0; x < gridSize.x; x++)
            {
                Vector3 spawnPos = new Vector3(x * gridDim.x, 0, z * gridDim.y);
                GameObject obj = Instantiate(gridTile, transform);
                obj.transform.position = spawnPos;
                obj.transform.rotation = Quaternion.identity;
                m_gridElements.Add(obj.transform, obj.GetComponent<GridComponent>());

            }
        }
    }
    private void Update()
    {
        if (Input.GetMouseButton(2))
        {
            Vector3 dir = (Input.mousePosition - m_pastMouse).normalized;

            m_cam.transform.position = m_cam.transform.position + (new Vector3(dir.x, 0, dir.y )*Time.deltaTime * moveSpeed);
            m_pastMouse = Input.mousePosition;
        }

        if(m_prevMouse != Input.mousePosition)
        {
            checkBlock();

            m_prevMouse = Input.mousePosition;
        }

        if (m_disableInput)
            return;

        if (Input.GetMouseButtonDown(0))
        {
            if (m_currentSelectedComponent != null)
            {
                m_currentSelectedComponent.SetBlockType(m_spawnBlockType);
            }
        }
        else if (Input.GetMouseButtonDown(1))
        {
            if (m_currentSelectedComponent != null)
            {
                m_currentSelectedComponent.VanishBlock(m_spawnBlockType);
            }
        }


    }
}
