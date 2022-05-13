using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

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
public enum GridType
{
    Selected,
    Painted,
    None
}
public enum PathType
{
    Straight,
    Diagonal
}
public enum Occupied
{
    Yes,
    No
}
public class GridManager : MonoBehaviour
{
    private struct Cost
    {
        public float Hcost;
        public float Gcost;
    }

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

    [SerializeField]
    private GameObject agent;
    [SerializeField]
    private int agentCount = 5;

    private Camera m_cam;

    private GridComponent m_currentSelectedComponent;

    private BlockType m_spawnBlockType = BlockType.Red;
    private GameMode m_gameMode = GameMode.Builder;
    private PathType m_pathType = PathType.Straight;

    private Vector3 m_prevMouse;
    private Dictionary<Vector3, GridComponent> m_gridElements;
    private bool m_disableInput = false;

    private Transform m_startPath;
    private Transform m_endPath;

    private Vector3 m_pastMouse;
    List<CombineInstance> m_redCombineMeshes;
    List<CombineInstance> m_blueCombineMeshes;
    public static GridManager Instance;

    private Dictionary<Agent, GridComponent> m_targetPoints;
    private Dictionary<Agent, GridComponent> m_startsPoints;
    private List<Agent> m_agents;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;

        }
        else
        {
            Destroy(gameObject);
        }

        m_cam = Camera.main;
        m_gridElements = new Dictionary<Vector3, GridComponent>();
        m_redCombineMeshes = new List<CombineInstance>();
        m_blueCombineMeshes = new List<CombineInstance>();

        m_targetPoints = new Dictionary<Agent, GridComponent>();
        m_startsPoints = new Dictionary<Agent, GridComponent>();
        m_agents = new List<Agent>();

        generateGrid();

    }
    private void OnEnable()
    {
        GameEvents.OnChooseObjectChangedEvent += onChooseObjectChanged;
        GameEvents.OnChooseModechangedEvent += onGameModeChanged;
        GameEvents.OnCoverUIEnterEvent += onCoverUIEnter;
        GameEvents.OnChoosePathTypechangedEvent += onChoosePathType;
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
        m_gameMode = gameMode;

        m_startsPoints.Clear();
        m_targetPoints.Clear();

        if (gameMode == GameMode.NonBuilder)
        {

            m_currentSelectedComponent = null;

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
                item.ReturnParent().SetActive(false);
                item.ClearGridTile();
                item.UnHoverBlock();
            }
            foreach (var item in m_agents)
            {
                item.gameObject.SetActive(true);
                item.StartJourney();
            }
        }
        else if (gameMode == GameMode.Builder)
        {
            foreach (var item in m_gridElements.Values)
            {
                item.ReturnParent().SetActive(true);
                item.ClearGridTile();
                item.UnHoverBlock();

            }
            redCombineMeshHolder.SetActive(false);
            blueCombineMeshHolder.SetActive(false);
            m_currentSelectedComponent = null;

            foreach (var item in m_agents)
            {
                item.gameObject.SetActive(false);
            }
        }
    }
    private void onChoosePathType(PathType type)
    {
        m_pathType = type;



        foreach (var item in m_gridElements.Values)
        {
            item.ClearGridTile();
        }

        foreach (var item in m_agents)
        {
            item.gameObject.SetActive(false);
        }

        m_startsPoints.Clear();
        m_targetPoints.Clear();

        foreach (var item in m_agents)
        {
            item.gameObject.SetActive(true);
            item.StartJourney();
        }
    }
    private void OnDisable()
    {
        GameEvents.OnChooseObjectChangedEvent -= onChooseObjectChanged;
        GameEvents.OnChooseModechangedEvent -= onGameModeChanged;
        GameEvents.OnCoverUIEnterEvent -= onCoverUIEnter;
        GameEvents.OnChoosePathTypechangedEvent -= onChoosePathType;

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
            if (m_gameMode == GameMode.Builder)
            {
                if (m_gridElements.TryGetValue(hit.transform.position, out info))
                {
                    if (info != m_currentSelectedComponent)
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
            else if (m_gameMode == GameMode.NonBuilder)
            {
                /*
                if (m_gridElements.TryGetValue(hit.transform.position, out info))
                {
                    if (info != m_currentSelectedComponent && info.GetBlockType()==BlockType.None)
                    {
                        if (m_currentSelectedComponent != null)
                        {
                            m_currentSelectedComponent.UnHoverGridTile();
                        }
                        info.HoverGridTile();
                        m_currentSelectedComponent = info;
                    }
                }
                */
            }
        }
    }
    private void Start()
    {
        for (int i = 0; i < agentCount; i++)
        {
            Agent ag = Instantiate(agent).GetComponent<Agent>();
          
            if (ag != null)
            {
                ag.gameObject.SetActive(false);
                m_agents.Add(ag);
            }
        }
    }
    private void generateGrid()
    {
        for (int z = 0; z < gridSize.y; z++)
        {
            for (int x = 0; x < gridSize.x; x++)
            {
                Vector3 spawnPos = new Vector3( x * gridDim.x, 0, z * gridDim.y);
                GameObject obj = Instantiate(gridTile, transform);
                obj.transform.position = spawnPos;
                obj.transform.rotation = Quaternion.identity;
                m_gridElements.Add(obj.transform.position, obj.GetComponent<GridComponent>());

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
            if (m_gameMode == GameMode.Builder)
            {
                if (m_currentSelectedComponent != null)
                {
                    m_currentSelectedComponent.SetBlockType(m_spawnBlockType);
                }
            }
            else if (m_gameMode == GameMode.NonBuilder)
            {
                /*
                if (m_currentSelectedComponent == null)
                    return;

                if (m_startPath == null)
                {
                    m_startPath = m_currentSelectedComponent.transform;
                    m_currentSelectedComponent.SelectGridTile();
                }
                else if (m_startPath != null && m_endPath == null)
                {
                    foreach (var item in m_gridElements.Values)
                    {
                        item.ClearGridTile();
                    }

                    m_endPath = m_currentSelectedComponent.transform;
                    m_currentSelectedComponent.SelectGridTile();
                    FindPath(m_startPath,m_endPath);
                }
                else if (m_startPath != null && m_endPath != null)
                {
                    foreach (var item in m_gridElements.Values)
                    {
                        item.ClearGridTile();
                    }

                    GridComponent grid;
                    if (m_gridElements.TryGetValue(m_startPath.position, out grid))
                    {
                        grid.ClearGridTile();
                    }
                    m_startPath = m_endPath;
                    m_endPath = m_currentSelectedComponent.transform;
                    m_currentSelectedComponent.SelectGridTile();
                    FindPath(m_startPath, m_endPath);
                }
                */

            }
            
        }
        else if (Input.GetMouseButtonDown(1))
        {
            if (m_gameMode == GameMode.Builder)
            {
                if (m_currentSelectedComponent != null)
                {
                    m_currentSelectedComponent.VanishBlock(m_spawnBlockType);
                }
            }
            
        }


    }
    private List<GridComponent> find8Neighbours(Vector3 pos)
    {
        List<GridComponent> cmpts = new List<GridComponent>();

        for (int z = -1; z <= 1; z++)
        {
            for (int x = -1; x <= 1; x++)
            {
                if (x != 0 || z != 0)
                {
                    Vector3 n = new Vector3(x * gridDim.x, 0, z * gridDim.y) + pos;

                    GridComponent grid;
                    if (m_gridElements.TryGetValue(n, out grid))
                    {
                        if (grid.GetBlockType() == BlockType.None)
                        {
                            cmpts.Add(grid);
                        }
                       
                    }
                }
                
            }
        }
        return cmpts;
    }
    private List<GridComponent> find4Neighbours(Vector3 pos)
    {
        List<GridComponent> cmpts = new List<GridComponent>();

        for (int z = -1; z <= 1; z++)
        {
            for (int x = -1; x <= 1; x++)
            {
                if ((x == 0 && z == -1) || (x == -1 && z == 0)  || (x == 1 && z == 0) || ((x == 0 && z == 1)))
                {
                    Vector3 n = new Vector3(x * gridDim.x, 0, z * gridDim.y) + pos;

                    GridComponent grid;
                    if (m_gridElements.TryGetValue(n, out grid))
                    {
                        if (grid.GetBlockType() == BlockType.None)
                        {
                            cmpts.Add(grid);
                        }

                    }
                }

            }
        }
        return cmpts;
    }

    public List<GridComponent> FindPath(Transform start, Transform end)
    {
        
        List<GridComponent> m_path = new List<GridComponent>();

        Dictionary<GridComponent, GridComponent> cameFrom = new Dictionary<GridComponent, GridComponent>();
        Dictionary<GridComponent, Cost> m_prevNodes = new Dictionary<GridComponent, Cost>(); ;
        Dictionary<GridComponent, bool> m_visitedNodes = new Dictionary<GridComponent, bool>();

        GridComponent stComp;
        if(m_gridElements.TryGetValue(start.position,out stComp))
        {
            Cost cost;
            cost.Gcost = 0;
            cost.Hcost = Vector3.Distance(stComp.transform.position, end.position);
            m_prevNodes.Add(stComp, cost);

        }
        else
        {
            return m_path;
        }
        while (m_prevNodes.Count > 0)
        {
            var current = m_prevNodes.OrderBy(item => item.Value.Hcost + item.Value.Gcost
            ).ToList()[0].Key;

            if (current == null)
            {
                break;
            }
            if (current.transform == end)
            {
                //Debug.Log("yo baby"+ current.transform.position);
                //Debug.Log("Camefrom "+ cameFrom.Count);
                if (current != null)
                {
                    m_path.Add(current);
                    while (cameFrom.ContainsKey(current))
                    {
                        current = cameFrom[current];
                        m_path.Add(current);
                    }
                    m_path.Reverse();
                }
                m_gridElements[start.position].PaintGridTile();
                foreach (var item in m_path)
                {
                    item.PaintGridTile();
                }
                return m_path;
            }
            List<GridComponent> neighbours = m_pathType == PathType.Diagonal
                ? find8Neighbours(current.transform.position)
                : find4Neighbours(current.transform.position);
            
            foreach (var item in neighbours)
            {
                if (m_prevNodes.ContainsKey(item))
                {
                    Cost cost;
                    if(m_prevNodes[item].Gcost > m_prevNodes[current].Gcost + 1)
                    {
                        cost.Gcost = m_prevNodes[current].Gcost + 1;
                        cameFrom[item] = current;
                    }
                    else
                    {
                        cost.Gcost = m_prevNodes[item].Gcost;
                    }

                    cost.Hcost = m_prevNodes[item].Hcost;

                    m_prevNodes[item] = cost;
                }
                else if(!m_prevNodes.ContainsKey(item))
                {
                    if (!m_visitedNodes.ContainsKey(item))
                    {
                        Cost cost;
                        cost.Gcost = m_prevNodes[current].Gcost + 1;
                        cost.Hcost = Vector3.Distance(item.transform.position, end.position);
                        cameFrom[item] = current;
                        m_prevNodes.Add(item, cost);
                    }
                }
            }
            m_visitedNodes.Add(current, true);
            m_prevNodes.Remove(current);
        }

        return null;
    }
    public GridComponent FindRandomTarget(Agent agent)
    {
        while (true)
        {
            int x = Random.Range(0, (int)gridSize.x);
            int z = Random.Range(0, (int)gridSize.y);

            Vector3 n = new Vector3(x * gridDim.x, 0, z * gridDim.y);
            GridComponent info;
            if (m_gridElements.TryGetValue(n, out info))
            {
                if (info.GetBlockType() == BlockType.None 
                    && !m_targetPoints.ContainsKey(agent))
                {
                    if (!m_targetPoints.ContainsValue(info))
                    {
                        m_targetPoints.Add(agent, info);
                        return info;
                    }
                }
                else if (info.GetBlockType() == BlockType.None 
                    && m_targetPoints.ContainsKey(agent))
                {
                    if (!m_targetPoints.ContainsValue(info))
                    {
                        m_targetPoints[agent] = info;
                        return info;
                    }
                }
            }
        }
    }
    public GridComponent FindRandomStart(Agent agent)
    {
        while (true)
        {
            int x = Random.Range(0, (int)gridSize.x);
            int z = Random.Range(0, (int)gridSize.y);

            Vector3 n = new Vector3(x * gridDim.x, 0, z * gridDim.y);
            GridComponent info;
            if (m_gridElements.TryGetValue(n, out info))
            {
                if (info.GetBlockType() == BlockType.None 
                    && !m_startsPoints.ContainsKey(agent))
                {
                    if (!m_startsPoints.ContainsValue(info))
                    {
                        m_startsPoints.Add(agent, info);
                        return info;
                    }
                }
                else if (info.GetBlockType() == BlockType.None 
                    && m_startsPoints.ContainsKey(agent))
                {
                    if (!m_startsPoints.ContainsValue(info))
                    {
                        m_startsPoints[agent] = info;
                        return info;
                    }
                }
            }
        }
    }

    
}
