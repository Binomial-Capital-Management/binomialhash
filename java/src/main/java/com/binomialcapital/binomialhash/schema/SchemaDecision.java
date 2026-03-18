package com.binomialcapital.binomialhash.schema;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Final type decision and scored alternatives for one column.
 */
public record SchemaDecision(
    String baseType,
    double confidence,
    Map<String, Double> candidateScores,
    List<String> semanticTags
) {
    public Map<String, Object> toDict() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("base_type", baseType);
        out.put("confidence", confidence);
        out.put("candidate_scores", candidateScores);
        out.put("semantic_tags", semanticTags);
        return out;
    }
}
