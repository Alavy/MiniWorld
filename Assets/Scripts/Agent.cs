using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent : MonoBehaviour
{
    private List<GridComponent> m_paths;

    private Transform m_startPath;
    private Transform m_endPath;
    private Coroutine m_routine;

    public void StartJourney()
    {
        GridComponent start = GridManager.Instance.FindRandomStart(this);
        m_startPath = start.transform;
        GridComponent end =  GridManager.Instance.FindRandomTarget(this);
        m_endPath = end.transform;

        if (m_startPath != null && m_endPath != null)
        {
            transform.position = m_startPath.position;
            m_routine = StartCoroutine(exploreGrid());
        }
    }

    IEnumerator exploreGrid()
    {
        while (true)
        {
            if (m_startPath != null && m_endPath != null)
            {
                transform.position = m_startPath.position;
                m_paths = GridManager.Instance.FindPath(m_startPath, m_endPath);
            }
            if (m_paths == null)
            {
                StopCoroutine(m_routine);
            }
            else
            {
                if (m_paths.Count > 0)
                {
                    //Debug.Log("Path -> " + m_paths.Count);

                    for (int i = 0; i < m_paths.Count; i++)
                    {
                        if (m_paths[i].Occupation() == Occupied.Yes)
                        {
                            yield return new WaitForSeconds(0.7f);
                        }
                        m_paths[i].SetOccupation();
                        transform.LeanMove(m_paths[i].transform.position, 0.5f);
                        yield return new WaitForSeconds(0.6f);
                        m_paths[i].FreeOccupation();
                    }
                    if (transform.position == m_endPath.position)
                    {
                        m_startPath = m_endPath;
                        GridComponent end = GridManager.Instance.FindRandomTarget(this);
                        m_endPath = end.transform;
                    }
                    else
                    {
                        StopCoroutine(m_routine);
                    }

                }
                yield return new WaitForSeconds(0.1f);
            }


        }

    }
}
