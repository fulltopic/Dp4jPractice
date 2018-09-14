package rl.dqn.supervised.fileprocess;

public class NTagProcessor {
    public static int VisitN(int position, int m) {
             if ((m >> 2 & 1) == 1) {
            // チー
                 System.out.println("Chow");
            return parseChow(position, m);
        } else if ((m >> 3 & 1) == 1) {
            // ポン
                 System.out.println("Pong");
            return parsePong(position, m);
        } else if ((m >> 4 & 1) == 1) {
            // 加カン
                 System.out.println("Kakan");
            return parseKakan(position, m);
//                 System.out.println("Not implemented kk");
//                 return -1;
        } else if ((m >> 5 & 1) == 1) {
                 System.out.println("Not implemented kita");
                 return -1;
            // 北
//            parseKita(position);
        } else if ((m & 3) == 0) {
                 System.out.println("ankan");
                 return parseAnkan(position, m);
            // 暗カン
//            parseAnkan(position, m);
        } else {
                 System.out.println("mikan");
                 // 明カン
                return parseMinkan(position, m);
        }
    }

    private static int parseChow(int position, int m) {
        int from = 3 - (m & 3); // 0: 上家, 1: 対面, 2: 下家, (3: 暗カンを表す)
        int tmp = (m >> 10) & 63;
        int r = tmp % 3; // 下から何番目の牌を鳴いたか

        tmp /= 3;
        tmp = tmp / 7 * 9 + tmp % 7;
        tmp *= 4; // 一番下の牌

        int[] h = new int[3];
        h[0] = tmp + ((m >> 3) & 3);
        h[1] = tmp + 4 + ((m >> 5) & 3);
        h[2] = tmp + 8 + ((m >> 7) & 3);

        int[] selfHai;
        int nakiHai;
        if (r == 0) {
            selfHai = new int[]{h[1], h[2]};
            nakiHai = h[0];
        } else if (r == 1) {
            selfHai = new int[]{h[0], h[2]};
            nakiHai = h[1];
        } else if (r == 2) {
            selfHai = new int[]{h[0], h[1]};
            nakiHai = h[2];
        } else {
            throw new RuntimeException();
        }

//        analyzer.chow(position, from, selfHai, nakiHai);
        for(int i = 0; i < selfHai.length; i ++) {
            System.out.println("SelfHai: " + selfHai[i]);
        }
        return nakiHai;
    }

    private static int parsePong(int position, int m) {
        int from = 3 - (m & 3); // 0: 上家, 1: 対面, 2: 下家, (3: 暗カンを表す)

        int unused = (m >> 5) & 3;
        int tmp = (m >> 9) & 127;
        int r = tmp % 3;

        tmp /= 3;
        tmp *= 4;

        int[] selfHai = new int[2];
        int count = 0;
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            if (i == unused) continue;
            if (count != r) {
                selfHai[idx++] = tmp + i;
            }
            count++;
        }
        int nakiHai = tmp + r;

//        analyzer.pong(position, from, selfHai, nakiHai);
        return nakiHai;
    }

    private static int parseKakan(int position, int m) {
        int from = 3 - (m & 3); // 0: 上家, 1: 対面, 2: 下家, (3: 暗カンを表す)
        int unused = (m >> 5) & 3;
        int tmp = (m >> 9) & 127;
        int r = tmp % 3;

        tmp /= 3;
        tmp *= 4;

        int[] selfHai = new int[2];
        int count = 0;
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            if (i == unused) continue;
            if (count != r) {
                selfHai[idx++] = tmp + i;
            }
            count++;
        }
        int nakiHai = tmp + r;
        int addHai = tmp + unused;

        for(int i = 0; i < selfHai.length; i ++) {
            System.out.println(selfHai[i]);
        }
//        analyzer.kakan(position, from, selfHai, nakiHai, addHai);

        return nakiHai;
    }

    private static int parseAnkan(int position, int m) {
        int tmp = (m >> 8) & 255;

        tmp = tmp / 4 * 4;

        int[] selfHai = {tmp + 1, tmp, tmp + 2, tmp + 3};

        for(int i = 0; i < selfHai.length; i ++) {
            System.out.println(selfHai[i]);
        }

        return tmp;
    }

    private static int parseMinkan(int position, int m) {
        int from = 3 - (m & 3); // 0: 上家, 1: 対面, 2: 下家, (3: 暗カンを表す)
        int nakiHai = (m >> 8) & 255; // 鳴いた牌

        int haiFirst = nakiHai / 4 * 4;

        int[] selfHai = new int[3];
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            if (haiFirst + idx == nakiHai) idx++;
            selfHai[i] = haiFirst + idx;
            idx++;
        }

//        analyzer.minkan(position, from, selfHai, nakiHai);
        for(int i = 0; i < selfHai.length; i ++) {
            System.out.println(selfHai[i]);
        }

        return nakiHai;
    }
}
