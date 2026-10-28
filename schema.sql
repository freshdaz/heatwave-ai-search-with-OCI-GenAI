-- Run in the same database as an existing WordPress wp_posts table.
-- VECTOR, STRING_TO_VECTOR(), and DISTANCE() require a compatible HeatWave service.
-- All stored embeddings and query embeddings must use the same model.
CREATE TABLE IF NOT EXISTS wp_posts_embeddings_cohere (
    ID BIGINT UNSIGNED NOT NULL,
    post_title TEXT NOT NULL,
    post_title_embedding VECTOR(1024) NOT NULL,
    post_excerpt TEXT NOT NULL,
    post_excerpt_embedding VECTOR(1024) NOT NULL,
    guid VARCHAR(255) NOT NULL DEFAULT '',
    post_summary TEXT NOT NULL,
    post_summary_embedding VECTOR(1024) NOT NULL,
    post_keywords JSON DEFAULT NULL,
    PRIMARY KEY (ID),
    FULLTEXT KEY title_excerpt_summary (post_title, post_excerpt, post_summary),
    KEY idx_post_keywords ((CAST(post_keywords AS CHAR(255) ARRAY)))
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
