import java.util.*;

public class patternlock {

    static Map<Character, List<Character>> adjacency = new HashMap<>();
    static {
        adjacency.put('A', Arrays.asList('B', 'D', 'E'));
        adjacency.put('B', Arrays.asList('A', 'C', 'D', 'E', 'F'));
        adjacency.put('C', Arrays.asList('B', 'E', 'F'));
        adjacency.put('D', Arrays.asList('A', 'B', 'E', 'G', 'H'));
        adjacency.put('E', Arrays.asList('A', 'B', 'C', 'D', 'F', 'G', 'H', 'I'));
        adjacency.put('F', Arrays.asList('B', 'C', 'E', 'H', 'I'));
        adjacency.put('G', Arrays.asList('D', 'E', 'H'));
        adjacency.put('H', Arrays.asList('D', 'E', 'F', 'G', 'I'));
        adjacency.put('I', Arrays.asList('E', 'F', 'H'));
    }

    static Map<List<Character>, Character> indirectPaths = new HashMap<>();
    static {
        indirectPaths.put(Arrays.asList('A', 'C'), 'B');
        indirectPaths.put(Arrays.asList('A', 'G'), 'D');
        indirectPaths.put(Arrays.asList('A', 'I'), 'E');
        indirectPaths.put(Arrays.asList('B', 'H'), 'E');
        indirectPaths.put(Arrays.asList('C', 'G'), 'E');
        indirectPaths.put(Arrays.asList('C', 'I'), 'F');
        indirectPaths.put(Arrays.asList('D', 'F'), 'E');
        indirectPaths.put(Arrays.asList('G', 'I'), 'H');
    }

    public static boolean isValidPath(List<Character> path) {
        Set<Character> visited = new HashSet<>();
        for (int i = 0; i < path.size() - 1; i++) {
            char current = path.get(i);
            char nextLabel = path.get(i + 1);

            if (visited.contains(nextLabel)) {
                continue;
            }

            if (!adjacency.get(current).contains(nextLabel)) {
                List<Character> key = Arrays.asList(current, nextLabel);
                Character requiredNode = indirectPaths.getOrDefault(key, null);
                if (requiredNode == null) {
                    requiredNode = indirectPaths.getOrDefault(Arrays.asList(nextLabel, current), null);
                }
                if (requiredNode != null && !visited.contains(requiredNode)) {
                    return false;
                }
            }
            visited.add(current);
        }
        return true;
    }

    public static List<List<Character>> permute(List<Character> arr, int length) {
        if (length == 1) {
            List<List<Character>> singleList = new ArrayList<>();
            for (Character c : arr) {
                singleList.add(Collections.singletonList(c));
            }
            return singleList;
        }

        List<List<Character>> perms = new ArrayList<>();
        for (int i = 0; i < arr.size(); i++) {
            Character first = arr.get(i);
            List<Character> remaining = new ArrayList<>(arr);
            remaining.remove(i);

            for (List<Character> perm : permute(remaining, length - 1)) {
                List<Character> subList = new ArrayList<>();
                subList.add(first);
                subList.addAll(perm);
                perms.add(subList);
            }
        }
        return perms;
    }

    public static List<String> listPatterns(char startLabel, char middleLabel, char endLabel) {
        List<Character> labels = Arrays.asList('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I');
        int length = 7;

        if (!labels.contains(startLabel) || !labels.contains(endLabel) || !labels.contains(middleLabel)) {
            throw new IllegalArgumentException("Start, end, or middle label is not in the list of labels.");
        }
        if (startLabel == endLabel) {
            throw new IllegalArgumentException("Start and end labels cannot be the same.");
        }
        if (middleLabel == startLabel || middleLabel == endLabel) {
            throw new IllegalArgumentException("Middle label cannot be the same as start or end labels.");
        }

        List<Character> remainingLabels = new ArrayList<>(labels);
        remainingLabels.remove(Character.valueOf(startLabel));
        remainingLabels.remove(Character.valueOf(endLabel));
        remainingLabels.remove(Character.valueOf(middleLabel));

        List<List<Character>> patterns = new ArrayList<>();
        List<List<Character>> permutations = permute(remainingLabels, length - 3);

        for (List<Character> perm : permutations) {
            for (int i = 0; i <= length - 3; i++) {
                List<Character> pattern = new ArrayList<>();
                pattern.add(startLabel);
                pattern.addAll(perm.subList(0, i));
                pattern.add(middleLabel);
                pattern.addAll(perm.subList(i, perm.size()));
                pattern.add(endLabel);
                if (isValidPath(pattern)) {
                    patterns.add(pattern);
                }
            }
        }

        List<String> patternList = new ArrayList<>();
        for (List<Character> pattern : patterns) {
            StringBuilder sb = new StringBuilder();
            for (Character ch : pattern) {
                sb.append(ch);
            }
            patternList.add(sb.toString());
        }

        return patternList;
    }

    public static void main(String[] args) {
        char startLabel = 'A';
        char middleLabel = 'I';
        char endLabel = 'C';

        List<String> patterns = listPatterns(startLabel, middleLabel, endLabel);
        System.out.println("Patterns starting with '" + startLabel + "', including '" + middleLabel + "', and ending with '" + endLabel + "':");
        System.out.println(patterns);
    }
}
