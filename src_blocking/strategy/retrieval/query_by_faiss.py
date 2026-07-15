import os
import logging
import numpy as np
import faiss

from src_blocking.model.evidence import RetrievalEvidence, AugmentationEvidence
from src_blocking.strategy.es_helper import determine_es_index_name
from src_blocking.strategy.retrieval.retrieval_strategy import RetrievalStrategy
from src_blocking.strategy.indexing.preprocess_records_and_index_es import prepare_processed_action


class QueryByFaiss(RetrievalStrategy):
    def __init__(self, dataset, embedding_dir, faiss_index_path,
                 clusters=False, switched=False,
                 similarity="cos"):
        super().__init__(dataset, "query_by_faiss", clusters=clusters, switched=switched)

        logger = logging.getLogger()
        logger.info("Initializing QueryByFaiss")

        self.embedding_dir = embedding_dir
        self.faiss_index_path = faiss_index_path
        self.switched = switched
        self.similarity = similarity

        # Load tableA embeddings
        self.emb_a = np.load(os.path.join(embedding_dir, "tableA_embeddings.npy")).astype("float32")
        logger.info(f"Loaded tableA embeddings: {self.emb_a.shape}")

        # Load FAISS index (built on tableB)
        self.index = faiss.read_index(faiss_index_path)
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")

        # If you used cosine via inner product, normalize queries
        self.normalize = (self.similarity == "cos")

    def retrieve_evidence(self, query_table, evidence_count, entity_id=None):
        logger = logging.getLogger()
        evidences = []
        evidence_id = 1

        # Query IDs from QueryTable (SC-Block stores them in row["entityId"])
        if entity_id is None:
            query_ids = [row["entityId"] for row in query_table.table]
        else:
            # keep order consistent with query_table.table for correct mapping
            query_ids = [row["entityId"] for row in query_table.table if row["entityId"] == entity_id]

        if len(query_ids) == 0:
            return []

        # Build query vectors from precomputed embeddings
        query_vecs = self.emb_a[query_ids].astype("float32")
        if self.normalize:
            faiss.normalize_L2(query_vecs)

        # FAISS search
        D, I = self.index.search(query_vecs, evidence_count)

        all_neighbor_ids = set(I.flatten().tolist())

        logger.info(f"Total unique neighbor IDs: {len(all_neighbor_ids)}")

        index_name = determine_es_index_name(
            self.schema_org_class,
            clusters=self.clusters,
            switched=self.switched
        )

        # ---------------------------------------------------------
        # 5) BULK fetch from ES or pandas
        # ---------------------------------------------------------
        entity_result = self.query_tables_index_by_id(
            list(all_neighbor_ids),
            index_name
        )

        if self._es is not None:
            hits = [
                prepare_processed_action(hit)
                for hit in entity_result["hits"]["hits"]
            ]
        else:
            hits = entity_result

        # Build ID → hit mapping (O(1) lookup later)
        id_to_hit = {int(hit["id"]): hit for hit in hits}

        logger.info(f"Fetched {len(id_to_hit)} candidate records")

        # ---------------------------------------------------------
        # 6) Build evidences (pure in-memory, no IO)
        # ---------------------------------------------------------
        for qi, qid in enumerate(query_ids):

            neighbor_ids = I[qi]
            distances = D[qi]

            # Create fast similarity lookup
            neighbor_map = {
                int(inst): float(dist)
                for inst, dist in zip(neighbor_ids, distances)
            }

            for inst in neighbor_ids:

                inst = int(inst)
                hit = id_to_hit.get(inst)

                if hit is None:
                    continue

                found_value = None
                if (
                    query_table.type == "augmentation"
                    and query_table.target_attribute in hit
                ):
                    found_value = hit[query_table.target_attribute]

                rowId = hit["row_id"]
                table_name = hit["table"]

                if query_table.type == "retrieval":
                    evidence = RetrievalEvidence(
                        evidence_id,
                        query_table.identifier,
                        qid,
                        table_name,
                        rowId,
                        hit
                    )
                elif query_table.type == "augmentation":
                    evidence = AugmentationEvidence(
                        evidence_id,
                        query_table.identifier,
                        qid,
                        table_name,
                        rowId,
                        hit,
                        found_value,
                        query_table.target_attribute
                    )
                else:
                    raise ValueError(
                        f"Query Table Type {query_table.type} is not defined!"
                    )

                evidence.scores[self.name] = neighbor_map[inst]
                evidence.similarity_score = neighbor_map[inst]

                evidences.append(evidence)
                evidence_id += 1

        logger.info(
            f"Retrieved {len(evidences)} evidences "
            f"for query table {query_table.identifier}"
        )

        return evidences
